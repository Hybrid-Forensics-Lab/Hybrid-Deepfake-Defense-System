"""
Evaluate ASR of robustness-transformed cloaked images against ArcFace or FaceNet.

For each transformed cloaked image:
  - Load the corresponding original clean image (arcface_path from manifest)
  - Compute embeddings for both
  - ASR = 1 if cosine_sim < 0.28 (ArcFace) or L2_dist > 1.1 (FaceNet)
  - Also compute SSIM and PSNR vs original clean image

Usage:
  python eval_robustness_asr.py \\
    --transform_dir  results/phase2/robustness/jpeg75 \\
    --manifest       data/lfw/manifest.csv \\
    --recognizer     arcface \\
    --arcface_weights weights/ms1mv3_arcface_r100_fp16.pth \\
    --batch_size     32 \\
    --output         results/phase2/robustness/jpeg75_arcface_asr.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ARCFACE_NORMALIZE_MEAN = [0.5, 0.5, 0.5]
ARCFACE_NORMALIZE_STD  = [0.5, 0.5, 0.5]
FACENET_NORMALIZE_MEAN = [0.5, 0.5, 0.5]
FACENET_NORMALIZE_STD  = [0.5, 0.5, 0.5]

ARCFACE_SIZE    = 112
FACENET_SIZE    = 160
ARCFACE_ASR_THRESH = 0.28   # cosine_sim < this → cloaked
FACENET_ASR_THRESH = 1.1    # L2_dist > this → cloaked


def fix_path(raw):
    p = str(raw).replace("\\", "/")
    if p.startswith("Hybrid-Deepfake-Defense-System/"):
        return REPO_ROOT / p[len("Hybrid-Deepfake-Defense-System/"):]
    if Path(p).is_absolute():
        return Path(p)
    return REPO_ROOT / p


def load_manifest(manifest_path):
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_stem_index(rows, path_key):
    """Map original image stem (e.g. 'Ahmed_Chalabi_0005') → manifest row."""
    index = {}
    for row in rows:
        raw = row[path_key]
        stem = Path(raw.replace("\\", "/")).stem
        index[stem] = row
    return index


def load_arcface_model(weights_path, device):
    from ml_privacy.models.arcface.iresnet import iresnet100
    backbone = iresnet100(pretrained=False, fp16=False)
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    backbone.load_state_dict(state, strict=False)
    return backbone.to(device).eval()


def load_facenet_model(device):
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model


def img_to_tensor(path, size):
    """Load image as RGB PIL, resize to (size, size), return float tensor [0, 1] shape [3, H, W]."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return TF.to_tensor(img)


def run_arcface_batches(model, pairs, batch_size, device):
    """
    pairs: list of (orig_path, cloaked_path)
    Returns list of (cosine_sim, asr) in same order.
    Uses fp16 autocast for inference.
    """
    mean = torch.tensor(ARCFACE_NORMALIZE_MEAN).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(ARCFACE_NORMALIZE_STD).view(1, 3, 1, 1).to(device)

    results = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="ArcFace inference"):
        batch = pairs[i : i + batch_size]

        orig_tensors    = torch.stack([img_to_tensor(o, ARCFACE_SIZE) for o, _ in batch])
        cloaked_tensors = torch.stack([img_to_tensor(c, ARCFACE_SIZE) for _, c in batch])

        orig_tensors    = orig_tensors.to(device)
        cloaked_tensors = cloaked_tensors.to(device)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                orig_raw    = model((orig_tensors    - mean) / std)
                cloaked_raw = model((cloaked_tensors - mean) / std)

        orig_emb    = F.normalize(orig_raw.float(),    p=2, dim=-1)
        cloaked_emb = F.normalize(cloaked_raw.float(), p=2, dim=-1)
        cos_sims = (orig_emb * cloaked_emb).sum(dim=-1).cpu().tolist()

        for sim in cos_sims:
            results.append((float(sim), 1 if sim < ARCFACE_ASR_THRESH else 0))

    return results


def run_facenet_batches(model, pairs, batch_size, device):
    """
    pairs: list of (orig_path, cloaked_path)
    For FaceNet: resize to 160×160, normalize with [0.5/0.5], compute L2 dist.
    Returns list of (l2_dist, asr) in same order.
    """
    mean = torch.tensor(FACENET_NORMALIZE_MEAN).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(FACENET_NORMALIZE_STD).view(1, 3, 1, 1).to(device)

    results = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="FaceNet inference"):
        batch = pairs[i : i + batch_size]

        orig_tensors    = torch.stack([img_to_tensor(o, FACENET_SIZE) for o, _ in batch])
        cloaked_tensors = torch.stack([img_to_tensor(c, FACENET_SIZE) for _, c in batch])

        orig_tensors    = orig_tensors.to(device)
        cloaked_tensors = cloaked_tensors.to(device)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                orig_raw    = model((orig_tensors    - mean) / std)
                cloaked_raw = model((cloaked_tensors - mean) / std)

        # FaceNet outputs are already L2-normalised by the model's last layer
        orig_emb    = orig_raw.float()
        cloaked_emb = cloaked_raw.float()
        l2_dists = (orig_emb - cloaked_emb).norm(p=2, dim=-1).cpu().tolist()

        for dist in l2_dists:
            results.append((float(dist), 1 if dist > FACENET_ASR_THRESH else 0))

    return results


def compute_ssim_psnr(orig_path, cloaked_path):
    orig_np    = np.array(Image.open(orig_path).convert("RGB"))
    cloaked_np = np.array(Image.open(cloaked_path).convert("RGB"))
    if orig_np.shape != cloaked_np.shape:
        cloaked_np = np.array(
            Image.fromarray(cloaked_np).resize(
                (orig_np.shape[1], orig_np.shape[0]), Image.LANCZOS
            )
        )
    orig_f    = orig_np.astype(np.float64) / 255.0
    cloaked_f = cloaked_np.astype(np.float64) / 255.0
    ssim_val  = structural_similarity(orig_f, cloaked_f, channel_axis=2, data_range=1.0)
    psnr_val  = peak_signal_noise_ratio(orig_f, cloaked_f, data_range=1.0)
    return float(ssim_val), float(psnr_val)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR of robustness-transformed cloaked images"
    )
    parser.add_argument("--transform_dir", type=Path, required=True,
                        help="Directory of transformed cloaked images (one transform)")
    parser.add_argument("--manifest", type=Path,
                        default=REPO_ROOT / "data/lfw/manifest.csv")
    parser.add_argument("--recognizer", choices=["arcface", "facenet"], required=True)
    parser.add_argument("--arcface_weights", type=Path,
                        default=REPO_ROOT / "ml_privacy/models/arcface/ms1mv3_arcface_r100_fp16.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Recognizer: {args.recognizer}")
    print(f"Transform dir: {args.transform_dir}")

    # Load manifest and build stem → row index keyed by the appropriate path column
    rows = load_manifest(args.manifest)
    path_key = "arcface_path" if args.recognizer == "arcface" else "facenet_path"
    stem_index = build_stem_index(rows, path_key)
    print(f"Manifest loaded: {len(rows)} rows, {len(stem_index)} unique {path_key} stems")

    # Collect transformed cloaked files — pattern depends on recognizer
    all_files = sorted(args.transform_dir.glob("*.png"))
    if args.recognizer == "arcface":
        cloaked_files = [f for f in all_files if f.name.endswith("_cloaked.png") and "facenet" not in f.name]
        cloaked_suffix = "_cloaked"
    else:
        cloaked_files = [f for f in all_files if f.name.endswith("_facenet_cloaked.png")]
        cloaked_suffix = "_facenet_cloaked"
    print(f"Transformed cloaked images: {len(cloaked_files)}")

    # Match each cloaked file to its original via manifest
    pairs = []       # (identity_id, orig_path, cloaked_path)
    missing = 0
    for cf in cloaked_files:
        stem = cf.stem  # e.g. "Ahmed_Chalabi_0005_cloaked" or "Ahmed_Chalabi_0005_facenet_cloaked"
        if stem.endswith(cloaked_suffix):
            orig_stem = stem[: -len(cloaked_suffix)]
        else:
            orig_stem = stem

        row = stem_index.get(orig_stem)
        if row is None:
            print(f"  WARN: no manifest entry for stem '{orig_stem}' ({cf.name})")
            missing += 1
            continue

        orig_path = fix_path(row[path_key])
        if not orig_path.exists():
            print(f"  WARN: original not found: {orig_path}")
            missing += 1
            continue

        pairs.append((row["identity_id"], orig_path, cf))

    print(f"Matched pairs: {len(pairs)}, missing: {missing}")
    if not pairs:
        raise RuntimeError("No pairs matched — check transform_dir and manifest.")

    orig_paths    = [p[1] for p in pairs]
    cloaked_paths = [p[2] for p in pairs]
    infer_pairs   = list(zip(orig_paths, cloaked_paths))

    # Run model inference
    if args.recognizer == "arcface":
        model = load_arcface_model(args.arcface_weights, device)
        print(f"ArcFace model loaded from {args.arcface_weights}")
        sim_results = run_arcface_batches(model, infer_pairs, args.batch_size, device)
        del model
        torch.cuda.empty_cache()
        metric_name = "cosine_sim"
    else:
        model = load_facenet_model(device)
        print("FaceNet (vggface2) loaded")
        sim_results = run_facenet_batches(model, infer_pairs, args.batch_size, device)
        del model
        torch.cuda.empty_cache()
        metric_name = "l2_dist"

    # Compute SSIM/PSNR vs original (arcface 112×112)
    ssim_psnr = []
    for orig_p, cloaked_p in tqdm(infer_pairs, desc="SSIM/PSNR"):
        try:
            s, p = compute_ssim_psnr(orig_p, cloaked_p)
        except Exception as e:
            print(f"  WARN: SSIM/PSNR error {cloaked_p}: {e}")
            s, p = float("nan"), float("nan")
        ssim_psnr.append((s, p))

    # Build output CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["identity_id", "image_path", metric_name, "asr", "ssim", "psnr"],
        )
        writer.writeheader()
        for (iid, _, cloaked_p), (metric_val, asr), (ssim_val, psnr_val) in zip(
            pairs, sim_results, ssim_psnr
        ):
            writer.writerow({
                "identity_id": iid,
                "image_path":  str(cloaked_p),
                metric_name:   f"{metric_val:.6f}",
                "asr":         asr,
                "ssim":        f"{ssim_val:.6f}",
                "psnr":        f"{psnr_val:.4f}",
            })

    # Aggregate stats
    n = len(sim_results)
    mean_asr  = sum(r[1] for r in sim_results) / n
    valid_ssim = [s for s, _ in ssim_psnr if not (s != s)]  # exclude nan
    valid_psnr = [p for _, p in ssim_psnr if not (p != p)]
    mean_ssim = float(np.mean(valid_ssim)) if valid_ssim else float("nan")
    mean_psnr = float(np.mean(valid_psnr)) if valid_psnr else float("nan")

    print(f"\n{'=' * 50}")
    print(f"  Recognizer : {args.recognizer}")
    print(f"  Transform  : {args.transform_dir.name}")
    print(f"  Images     : {n}")
    print(f"  Mean ASR   : {mean_asr:.4f}")
    print(f"  Mean SSIM  : {mean_ssim:.4f}")
    print(f"  Mean PSNR  : {mean_psnr:.2f} dB")
    print(f"{'=' * 50}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
