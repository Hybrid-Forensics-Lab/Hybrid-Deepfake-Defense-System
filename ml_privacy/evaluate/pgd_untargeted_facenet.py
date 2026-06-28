import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

FACENET_SIZE = 160
NORMALIZE = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# FaceNet verification threshold: Euclidean distance > 1.1 = identities differ
ASR_THRESHOLD = 1.1


def fix_path(raw):
    p = str(raw).replace("\\", "/")
    if p.startswith("Hybrid-Deepfake-Defense-System/"):
        return REPO_ROOT / p[len("Hybrid-Deepfake-Defense-System/"):]
    if Path(p).is_absolute():
        return Path(p)
    return REPO_ROOT / p


def load_facenet(device):
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    class FaceNetWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(NORMALIZE(x))

    return FaceNetWrapper(model)


def pgd_untargeted_batch(model, x_orig, epsilon, steps, alpha):
    """
    Batched untargeted PGD: maximises L2 embedding distance from the original.
    Operates in fp32. x_orig: [B, 3, 160, 160] on device, values in [0, 1].
    Returns x_adv on CPU.
    """
    x_orig = x_orig.detach()
    with torch.no_grad():
        e_orig = model(x_orig)  # [B, 512]

    # Random start within L-inf ball
    delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    delta = ((x_orig + delta).clamp(0, 1) - x_orig)
    x_adv = (x_orig + delta).clamp(0, 1).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        e_adv = model(x_adv)
        # Maximise L2 distance: gradient ascent on L2 norm
        loss = torch.linalg.vector_norm(e_adv - e_orig.detach(), ord=2, dim=-1).sum()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = (x_adv - x_orig).clamp(-epsilon, epsilon)
            x_adv = (x_orig + delta).clamp(0, 1)

    return x_adv.detach().cpu()


def compute_ssim_psnr(orig_np, cloaked_np):
    orig_f = orig_np.astype(np.float64) / 255.0
    cloaked_f = cloaked_np.astype(np.float64) / 255.0
    ssim_val = structural_similarity(orig_f, cloaked_f, channel_axis=2, data_range=1.0)
    psnr_val = peak_signal_noise_ratio(orig_f, cloaked_f, data_range=1.0)
    return float(ssim_val), float(psnr_val)


def parse_args():
    p = argparse.ArgumentParser(description="Batched untargeted PGD against FaceNet.")
    p.add_argument("--manifest", type=Path, default=Path("data/lfw/manifest.csv"))
    p.add_argument("--epsilon", type=float, default=0.03)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--alpha_factor", type=float, default=2.0,
                   help="alpha = epsilon / steps * alpha_factor")
    p.add_argument("--n_images", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=Path, default=Path("results/phase2/privacy"))
    p.add_argument("--save_images", type=lambda x: x.lower() == "true", default=False,
                   help="Save cloaked PNGs to output_dir/cloaked_images/ (True for full run)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = args.epsilon / args.steps * args.alpha_factor
    print(f"Device: {device}")
    print(f"eps={args.epsilon}  steps={args.steps}  alpha={alpha:.5f}  batch={args.batch_size}")

    tag = f"eps{args.epsilon}_steps{args.steps}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / f"pgd_untargeted_facenet_{tag}_results.csv"
    metrics_csv = args.output_dir / f"pgd_untargeted_facenet_{tag}_metrics.csv"
    images_dir = args.output_dir / "cloaked_images" if args.save_images else None
    if images_dir:
        images_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    manifest = manifest.sample(n=min(args.n_images, len(manifest)),
                               random_state=args.seed).reset_index(drop=True)
    manifest["_path"] = manifest["facenet_path"].apply(fix_path)
    print(f"Manifest sampled: {len(manifest)} images")

    # Resume support
    done_paths = set()
    existing_rows = []
    if results_csv.exists():
        df_done = pd.read_csv(results_csv)
        done_paths = set(df_done["image_path"].tolist())
        existing_rows = df_done.to_dict("records")
        print(f"Resuming: {len(done_paths)} already processed")

    todo = manifest[~manifest["_path"].astype(str).isin(done_paths)].reset_index(drop=True)
    print(f"To process: {len(todo)}")

    transform = T.Compose([T.Resize((FACENET_SIZE, FACENET_SIZE)), T.ToTensor()])

    print("Loading FaceNet (InceptionResnetV1 vggface2)...")
    model = load_facenet(device)
    model.eval()

    new_rows = []
    batches = [todo.iloc[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]

    for batch_df in tqdm(batches, desc=f"PGD batches eps={args.epsilon}"):
        imgs, meta = [], []
        for _, row in batch_df.iterrows():
            try:
                img = Image.open(row["_path"]).convert("RGB")
                imgs.append(transform(img))
                meta.append(row)
            except Exception as e:
                print(f"Skip {row['_path']}: {e}")

        if not imgs:
            continue

        x_orig = torch.stack(imgs).to(device)
        x_adv = pgd_untargeted_batch(model, x_orig, args.epsilon, args.steps, alpha)

        with torch.no_grad():
            e_orig = model(x_orig)
            e_adv = model(x_adv.to(device))
            l2_dists = torch.linalg.vector_norm(e_orig - e_adv, ord=2, dim=-1).cpu().numpy()

        for i, (row, l2_dist) in enumerate(zip(meta, l2_dists)):
            orig_np = (x_orig[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            cloaked_np = (x_adv[i].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            ssim_val, psnr_val = compute_ssim_psnr(orig_np, cloaked_np)
            asr = int(l2_dist > ASR_THRESHOLD)

            if args.save_images and images_dir:
                fname = Path(str(row["_path"])).stem + "_facenet_cloaked.png"
                Image.fromarray(cloaked_np).save(images_dir / fname)

            new_rows.append({
                "identity_id": row["identity_id"],
                "image_path": str(row["_path"]),
                "l2_dist": float(l2_dist),
                "asr": asr,
                "ssim": float(ssim_val),
                "psnr": float(psnr_val),
            })

        pd.DataFrame(existing_rows + new_rows).to_csv(results_csv, index=False)

    df = pd.DataFrame(existing_rows + new_rows)
    mean_asr = df["asr"].mean()
    mean_ssim = df["ssim"].mean()
    mean_psnr = df["psnr"].mean()
    mean_l2 = df["l2_dist"].mean()

    print(f"\n{'='*52}")
    print(f"  epsilon={args.epsilon}  steps={args.steps}  n={len(df)}")
    print(f"  ASR     : {mean_asr:.4f}  ({int(df['asr'].sum())}/{len(df)} fooled)")
    print(f"  SSIM    : {mean_ssim:.4f}")
    print(f"  PSNR    : {mean_psnr:.2f} dB")
    print(f"  L2 dist : {mean_l2:.4f}  (threshold={ASR_THRESHOLD})")
    print(f"{'='*52}")

    pd.DataFrame({
        "epsilon": [args.epsilon],
        "steps": [args.steps],
        "n_images": [len(df)],
        "mean_asr": [mean_asr],
        "mean_ssim": [mean_ssim],
        "mean_psnr": [mean_psnr],
        "mean_l2_dist": [mean_l2],
    }).to_csv(metrics_csv, index=False)

    print(f"Results : {results_csv}")
    print(f"Metrics : {metrics_csv}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
