import sys
import re
import random
import argparse
import csv
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm


WANG2020_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# UnivFD uses CLIP stats, not ImageNet — matches univfd_inference.py exactly
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

UNIVFD_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def sample_images(cloaked_dir):
    cloaked_dir = Path(cloaked_dir)
    images = sorted(cloaked_dir.glob("*.png"))
    print(f"Found {len(images)} PNG files in {cloaked_dir}")

    pattern = re.compile(r'^(.+)_\d+$')
    identity_map = {}
    for img in images:
        m = pattern.match(img.stem)
        identity = m.group(1) if m else img.stem
        identity_map.setdefault(identity, []).append(img)

    print(f"Found {len(identity_map)} identities")
    random.seed(42)
    selected = []
    for identity in sorted(identity_map):
        chosen = random.choice(identity_map[identity])
        selected.append((identity, chosen))

    print(f"Sampled {len(selected)} images (1 per identity)")
    return selected


def run_wang2020(selected, model_dir, weights_path, device):
    sys.path.insert(0, str(Path(model_dir).resolve()))
    from networks.resnet import resnet50

    model = resnet50(num_classes=1)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Wang2020 loaded from {weights_path}")

    rows = []
    with torch.no_grad():
        for identity, img_path in tqdm(selected, desc="Wang2020"):
            img = Image.open(img_path).convert("RGB")
            tensor = WANG2020_TRANSFORM(img).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor)).squeeze().item()
            rows.append({
                "image_path": str(img_path),
                "identity": identity,
                "forensic_model": "Wang2020",
                "predicted_prob": prob,
                "flagged_as_synthetic": 1 if prob > 0.5 else 0,
            })

    del model
    torch.cuda.empty_cache()
    return rows


def run_univfd(selected, univfd_dir, weights_path, device):
    univfd_dir = Path(univfd_dir).resolve()
    sys.path.insert(0, str(univfd_dir))
    from models import get_model

    model = get_model("CLIP:ViT-L/14")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.fc.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"UnivFD loaded from {weights_path}")

    rows = []
    with torch.no_grad():
        for identity, img_path in tqdm(selected, desc="UnivFD"):
            img = Image.open(img_path).convert("RGB")
            tensor = UNIVFD_TRANSFORM(img).unsqueeze(0).to(device)
            prob = model(tensor).sigmoid().item()
            rows.append({
                "image_path": str(img_path),
                "identity": identity,
                "forensic_model": "UnivFD",
                "predicted_prob": prob,
                "flagged_as_synthetic": 1 if prob > 0.5 else 0,
            })

    del model
    torch.cuda.empty_cache()
    return rows


def main():
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument("--cloaked_dir", type=str, required=True,
                        help="Flat folder of cloaked PNG images")
    parser.add_argument("--output_dir", type=str,
                        default=str(repo_root / "results" / "conflict"),
                        help="Directory for output CSVs (default: results/conflict/)")
    parser.add_argument("--wang2020_model_dir", type=str,
                        default=str(repo_root / "ml_forensics" / "models" / "CNNDetection"))
    parser.add_argument("--wang2020_weights", type=str,
                        default=str(repo_root / "ml_forensics" / "models" / "CNNDetection"
                                    / "weights" / "blur_jpg_prob0.5.pth"))
    parser.add_argument("--univfd_dir", type=str,
                        default=str(repo_root / "ml_forensics" / "models" / "UniversalFakeDetect"))
    parser.add_argument("--univfd_weights", type=str, default=None,
                        help="UnivFD fc_weights.pth (default: <univfd_dir>/pretrained_weights/fc_weights.pth)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.univfd_weights is None:
        args.univfd_weights = str(Path(args.univfd_dir) / "pretrained_weights" / "fc_weights.pth")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    selected = sample_images(args.cloaked_dir)

    all_rows = []
    all_rows.extend(run_wang2020(selected, args.wang2020_model_dir, args.wang2020_weights, device))
    all_rows.extend(run_univfd(selected, args.univfd_dir, args.univfd_weights, device))

    per_image_csv = output_dir / "fpr_test_results.csv"
    fieldnames = ["image_path", "identity", "forensic_model", "predicted_prob", "flagged_as_synthetic"]
    with open(per_image_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({**row, "predicted_prob": f"{row['predicted_prob']:.6f}"})
    print(f"Per-image results saved to {per_image_csv}")

    results_df = pd.read_csv(per_image_csv)
    summary_rows = []
    for model_name, group in results_df.groupby("forensic_model"):
        num_images = len(group)
        num_flagged = int(group["flagged_as_synthetic"].sum())
        fpr = num_flagged / num_images
        summary_rows.append({
            "forensic_model": model_name,
            "num_images": num_images,
            "num_flagged": num_flagged,
            "fpr": fpr,
        })

    summary_csv = output_dir / "preliminary_fpr.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}")

    print("\n--- FPR Results ---")
    for row in summary_rows:
        print(f"{row['forensic_model']}: FPR = {row['fpr']:.4f}  ({row['num_flagged']}/{row['num_images']} flagged)")


if __name__ == "__main__":
    main()
