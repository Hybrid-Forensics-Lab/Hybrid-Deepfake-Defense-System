"""
Conflict matrix evaluation: measure FPR of forensic models on cloaked real images.

All images in --cloaked_dir are real (true_label=0). FPR = fraction flagged as synthetic.

Supported models:
  wang2020    -- ResNet-50 CNNDetection
  univfd_knn  -- CLIP ViT-L/14 + original fc head (k-NN proxy)
  ft_univfd   -- CLIP ViT-L/14 feature extraction + sklearn probe (.pkl)
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLIP_MEAN     = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD      = [0.26862954, 0.26130258, 0.27577711]


class ImageDirDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), str(self.paths[idx])


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_wang2020(model_dir, weights_path, device):
    sys.path.insert(0, str(model_dir))
    from networks.resnet import resnet50
    model = resnet50(num_classes=1)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


def load_univfd_knn(model_dir, weights_path, device):
    sys.path.insert(0, str(model_dir))
    from models import get_model
    model = get_model("CLIP:ViT-L/14")
    state = torch.load(weights_path, map_location="cpu")
    model.fc.load_state_dict(state)
    return model.to(device).eval()


def load_clip_backbone(device):
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    return clip_model, preprocess


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def infer_wang2020(model, loader, threshold, device):
    probs = []
    paths = []
    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(loader, desc="Wang2020"):
            batch_imgs = batch_imgs.to(device)
            p = torch.sigmoid(model(batch_imgs)).squeeze(1).cpu().tolist()
            probs.extend(p)
            paths.extend(batch_paths)
    return paths, probs


def infer_univfd_knn(model, loader, threshold, device):
    probs = []
    paths = []
    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(loader, desc="UnivFD k-NN"):
            batch_imgs = batch_imgs.to(device)
            p = model(batch_imgs).sigmoid().squeeze(1).cpu().tolist()
            probs.extend(p)
            paths.extend(batch_paths)
    return paths, probs


def infer_ft_univfd(clip_model, probe, image_paths, preprocess, batch_size, device):
    probs = []
    paths = []
    all_features = []

    dataset = ImageDirDataset(image_paths, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(loader, desc="FT-UnivFD (extract)"):
            batch_imgs = batch_imgs.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                feats = clip_model.encode_image(batch_imgs)
            feats = torch.nn.functional.normalize(feats.float(), p=2, dim=-1)
            all_features.append(feats.cpu().numpy())
            paths.extend(batch_paths)

    X = np.vstack(all_features)
    probs = probe.predict_proba(X)[:, 1].tolist()
    return paths, probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloaked_dir",  required=True,
                        help="Directory containing cloaked PNG images (all real, label=0)")
    parser.add_argument("--model",        required=True,
                        choices=["wang2020", "univfd_knn", "ft_univfd"])
    parser.add_argument("--weights",      default=None,
                        help="Path to model weights (.pth). Required for wang2020 and univfd_knn.")
    parser.add_argument("--model_dir",    default=None,
                        help="Path to model source code directory. Required for wang2020 and univfd_knn.")
    parser.add_argument("--probe",        default=None,
                        help="Path to sklearn probe .pkl file. Required for ft_univfd.")
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--output",       required=True,
                        help="Path to output metrics CSV.")
    parser.add_argument("--pattern",      default="*.png",
                        help="Glob pattern to filter images in cloaked_dir (default: *.png)")
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cloaked_dir = Path(args.cloaked_dir)
    image_paths = sorted(cloaked_dir.glob(args.pattern))
    if not image_paths:
        print(f"No images found in {cloaked_dir} matching '{args.pattern}'")
        return
    print(f"Found {len(image_paths)} cloaked images")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Wang2020 ---
    if args.model == "wang2020":
        model_dir   = Path(args.model_dir) if args.model_dir else REPO_ROOT / "ml_forensics/models/CNNDetection"
        weights_path = Path(args.weights) if args.weights else model_dir / "weights/blur_jpg_prob0.5.pth"

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        dataset = ImageDirDataset(image_paths, transform)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

        model = load_wang2020(model_dir, weights_path, device)
        print(f"Wang2020 weights loaded from {weights_path}")
        paths, probs = infer_wang2020(model, loader, args.threshold, device)
        del model
        torch.cuda.empty_cache()

    # --- UnivFD k-NN ---
    elif args.model == "univfd_knn":
        model_dir    = Path(args.model_dir) if args.model_dir else REPO_ROOT / "ml_forensics/models/UniversalFakeDetect"
        weights_path = Path(args.weights) if args.weights else model_dir / "pretrained_weights/fc_weights.pth"

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
        dataset = ImageDirDataset(image_paths, transform)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

        model = load_univfd_knn(model_dir, weights_path, device)
        print(f"UnivFD k-NN weights loaded from {weights_path}")
        paths, probs = infer_univfd_knn(model, loader, args.threshold, device)
        del model
        torch.cuda.empty_cache()

    # --- FT-UnivFD (CLIP + probe) ---
    elif args.model == "ft_univfd":
        probe_path = Path(args.probe) if args.probe else REPO_ROOT / "results/phase2/forensics/mlp_probe_10k.pkl"
        probe = joblib.load(probe_path)
        print(f"Probe loaded from {probe_path}")

        clip_model, preprocess = load_clip_backbone(device)
        print("CLIP ViT-L/14 backbone loaded")
        paths, probs = infer_ft_univfd(clip_model, probe, image_paths, preprocess,
                                       args.batch_size, device)
        del clip_model
        torch.cuda.empty_cache()

    # --- Compute FPR ---
    num_images  = len(probs)
    num_flagged = sum(1 for p in probs if p > args.threshold)
    fpr         = num_flagged / num_images
    passes_15pct = fpr < 0.15

    print(f"\n--- Conflict Matrix Result ---")
    print(f"Model:        {args.model}")
    print(f"Threshold:    {args.threshold}")
    print(f"Images:       {num_images}")
    print(f"Flagged:      {num_flagged}")
    print(f"FPR:          {fpr:.4f} ({fpr*100:.1f}%)")
    print(f"Passes <15%:  {passes_15pct}")

    metrics = pd.DataFrame([{
        "forensic_model":  args.model,
        "threshold":       args.threshold,
        "num_images":      num_images,
        "num_flagged":     num_flagged,
        "fpr":             round(fpr, 4),
        "passes_15pct":    passes_15pct,
    }])
    metrics.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")

    per_image = pd.DataFrame({
        "image_path":     [str(p) for p in paths],
        "predicted_prob": probs,
        "flagged":        [p > args.threshold for p in probs],
    })
    per_image_path = output_path.with_name(output_path.stem + "_per_image.csv")
    per_image.to_csv(per_image_path, index=False)
    print(f"Per-image    → {per_image_path}")


if __name__ == "__main__":
    main()
