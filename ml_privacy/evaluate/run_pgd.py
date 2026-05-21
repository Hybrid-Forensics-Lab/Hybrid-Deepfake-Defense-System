import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_privacy.attacks.pgd_attack import pgd_untargeted
from ml_privacy.attacks.targeted_pgd import pgd_targeted


ARCFACE_SIZE = 112
FACENET_SIZE = 160

# No normalization here — attack operates in [0, 1] pixel space.
# Each model wrapper normalizes internally before the backbone.
ARCFACE_TRANSFORM = T.Compose([
    T.Resize((ARCFACE_SIZE, ARCFACE_SIZE)),
    T.ToTensor(),
])

FACENET_TRANSFORM = T.Compose([
    T.Resize((FACENET_SIZE, FACENET_SIZE)),
    T.ToTensor(),
])


def load_arcface(device, weights_path):
    # insightface ONNX runtime has no autograd — use the PyTorch iresnet backbone.
    from ml_privacy.models.arcface.iresnet import iresnet100
    import torch.nn.functional as F

    backbone = iresnet100(pretrained=False, fp16=False)
    ckpt = torch.load(weights_path, map_location="cpu")
    # Checkpoints may be plain state_dict or wrapped under a key
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    # Strip 'module.' prefix from DataParallel checkpoints
    state = {k.replace("module.", ""): v for k, v in state.items()}
    backbone.load_state_dict(state, strict=False)

    _norm = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    class ArcFaceWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            # x: [0, 1] pixel space — normalize to [-1, 1] before backbone.
            return F.normalize(self.m(_norm(x)), p=2, dim=-1)

    return ArcFaceWrapper(backbone).to(device)


def load_facenet(device):
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    _norm = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    class FaceNetWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            # x: [0, 1] pixel space — normalize to [-1, 1] before backbone.
            return self.m(_norm(x))

    return FaceNetWrapper(model).to(device)


def load_image(path, transform, device):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def tensor_to_pil(t):
    # t: [0, 1] pixel space — attack operates and returns in this space.
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def compute_mean_embedding(paths, transform, model, device):
    embs = []
    model.eval()
    with torch.no_grad():
        for p in paths:
            try:
                img = load_image(p, transform, device)
                emb = model(img)
                embs.append(emb.squeeze(0))
            except Exception:
                continue
    if not embs:
        return None
    return torch.stack(embs).mean(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PGD cloaking on LFW corpus.")
    parser.add_argument("--manifest", type=Path, default=Path("data/lfw/manifest.csv"))
    parser.add_argument("--recognizer", choices=["arcface", "facenet"], default="facenet")
    parser.add_argument("--attack", choices=["untargeted", "targeted"], default="untargeted")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=Path("results/privacy/cloaked_images"))
    parser.add_argument(
        "--arcface-weights",
        type=Path,
        default=Path("ml_privacy/models/arcface/ms1mv3_arcface_r100_fp16.pth"),
        help="Path to iresnet100 pretrained .pth checkpoint (required when --recognizer arcface).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = pd.read_csv(args.manifest)
    img_col = "arcface_path" if args.recognizer == "arcface" else "facenet_path"
    transform = ARCFACE_TRANSFORM if args.recognizer == "arcface" else FACENET_TRANSFORM

    attack_dir = f"pgd_{args.attack}"
    out_path = args.output_dir / attack_dir / args.recognizer
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.recognizer}...")
    if args.recognizer == "arcface":
        if not args.arcface_weights.exists():
            raise FileNotFoundError(
                f"ArcFace weights not found at {args.arcface_weights}. "
                "Download ms1mv3_arcface_r100_fp16.pth from the insightface model zoo "
                "and pass --arcface-weights <path>."
            )
        model = load_arcface(device, args.arcface_weights)
    else:
        model = load_facenet(device)
    model.eval()

    decoy_embeddings = None
    # if args.attack == "targeted":
    #     identities = manifest["identity_id"].unique().tolist()
    #     assert len(identities) >= 2, "Need at least 2 identities for targeted attack."
    #     print("Pre-computing mean embeddings for all identities...")
    #     id_to_emb = {}
    #     for iid in tqdm(identities, desc="Mean embeddings"):
    #         rows = manifest[manifest["identity_id"] == iid]
    #         paths = [Path(p) for p in rows[img_col].tolist()]
    #         emb = compute_mean_embedding(paths, transform, model, device)
    #         if emb is not None:
    #             id_to_emb[iid] = emb
    #     decoy_pool = {k: v for k, v in id_to_emb.items()}
    if args.attack == "targeted":
            identities = manifest["identity_id"].unique().tolist()
            assert len(identities) >= 2, "Need at least 2 identities for targeted attack."
            print("Pre-computing mean embeddings for all identities...")
            id_to_emb = {}
            for iid in tqdm(identities, desc="Mean embeddings"):
                rows = manifest[manifest["identity_id"] == iid]
                
                # --- NEW PATH SANITIZATION HERE ---
                paths = []
                for p in rows[img_col].tolist():
                    raw_path = str(p).replace("\\", "/") 
                    if raw_path.startswith("Hybrid-Deepfake-Defense-System/"):
                        raw_path = raw_path.replace("Hybrid-Deepfake-Defense-System/", "", 1) 
                    paths.append(Path(raw_path))
                # ----------------------------------
                
                emb = compute_mean_embedding(paths, transform, model, device)
                if emb is not None:
                    id_to_emb[iid] = emb
            decoy_pool = {k: v for k, v in id_to_emb.items()}

    print(f"Running {args.attack} PGD (eps={args.epsilon}, step={args.step_size}, steps={args.num_steps})...")
    # for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Images"):
    #     iid = row["identity_id"]
    #     raw_path = str(row[img_col]).replace("\\", "/") # Convert Windows slashes to Unix
    #     if raw_path.startswith("Hybrid-Deepfake-Defense-System/"):
    #         raw_path = raw_path.replace("Hybrid-Deepfake-Defense-System/", "", 1) # Strip the duplicated root
    #     src_path = Path(raw_path)
    #     fname = src_path.stem + ".png"

    #     if out_path.exists():
    #         continue

    #     try:
    #         img_tensor = load_image(src_path, transform, device)
    #     except Exception as e:
    #         print(f"Skip {src_path}: {e}")
    #         continue

    #     if args.attack == "untargeted":
    #         cloaked = pgd_untargeted(
    #             img_tensor,
    #             model,
    #             epsilon=args.epsilon,
    #             step_size=args.step_size,
    #             num_steps=args.num_steps,
    #         )
    #     else:
    #         other_ids = [k for k in decoy_pool if k != iid]
    #         if not other_ids:
    #             print(f"No decoy for identity {iid}, skip.")
    #             continue
    #         decoy_id = random.choice(other_ids)
    #         decoy_emb = decoy_pool[decoy_id].unsqueeze(0).to(device)
    #         cloaked = pgd_targeted(
    #             img_tensor,
    #             model,
    #             decoy_embedding=decoy_emb,
    #             epsilon=args.epsilon,
    #             step_size=args.step_size,
    #             num_steps=args.num_steps,
    #         )

    #     pil_img = tensor_to_pil(cloaked)
    #     pil_img.save(out_path)

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Images"):
            iid = row["identity_id"]
            raw_path = str(row[img_col]).replace("\\", "/") 
            if raw_path.startswith("Hybrid-Deepfake-Defense-System/"):
                raw_path = raw_path.replace("Hybrid-Deepfake-Defense-System/", "", 1) 
            src_path = Path(raw_path)
            fname = src_path.stem + ".png"

            # 1. Define the actual file path for this specific image
            img_out_path = out_path / fname

            # 2. Check if the FILE exists, not the directory
            if img_out_path.exists():
                continue

            try:
                img_tensor = load_image(src_path, transform, device)
            except Exception as e:
                print(f"Skip {src_path}: {e}")
                continue

            if args.attack == "untargeted":
                cloaked = pgd_untargeted(
                    img_tensor,
                    model,
                    epsilon=args.epsilon,
                    step_size=args.step_size,
                    num_steps=args.num_steps,
                )
            else:
                other_ids = [k for k in decoy_pool if k != iid]
                if not other_ids:
                    print(f"No decoy for identity {iid}, skip.")
                    continue
                decoy_id = random.choice(other_ids)
                decoy_emb = decoy_pool[decoy_id].unsqueeze(0).to(device)
                cloaked = pgd_targeted(
                    img_tensor,
                    model,
                    decoy_embedding=decoy_emb,
                    epsilon=args.epsilon,
                    step_size=args.step_size,
                    num_steps=args.num_steps,
                )

            pil_img = tensor_to_pil(cloaked)
            
            # 3. Save to the specific file path
            pil_img.save(img_out_path)

    print(f"Done. Cloaked images saved to {out_path}")


if __name__ == "__main__":
    main()
