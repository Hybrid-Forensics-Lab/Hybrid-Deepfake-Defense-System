import sys
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class ManifestDataset(Dataset):
    def __init__(self, manifest_path, repo_root, transform):
        self.df = pd.read_csv(manifest_path)
        self.repo_root = Path(repo_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.repo_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, int(row["label"]), str(row["image_path"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repo-root", type=str, default=None,
                        help="Repo root for resolving image_path values in manifest. "
                             "Defaults to 4 levels up from manifest (FF++ layout).")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.model_dir).resolve()))
    from networks.resnet import resnet50

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.repo_root is not None:
        repo_root = Path(args.repo_root).resolve()
    else:
        # Default: 4 levels up from manifest (data/ff_plus_plus/256x256/manifest.csv)
        repo_root = Path(args.manifest).resolve().parents[3]

    # Match official eval: no resize (images already at loadSize=256), CenterCrop to 224
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ManifestDataset(args.manifest, repo_root, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Dataset size: {len(dataset)} images")

    model = resnet50(num_classes=1)
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Weights loaded from {args.weights}")

    all_paths = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for tensors, labels, paths in tqdm(loader, desc="Inference"):
            tensors = tensors.to(device)
            probs = torch.sigmoid(model(tensors)).squeeze(1).cpu().tolist()
            all_paths.extend(paths)
            all_labels.extend(labels.tolist())
            all_probs.extend(probs)

    results_path = output_dir / "wang2020_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "predicted_prob"])
        for path, label, prob in zip(all_paths, all_labels, all_probs):
            writer.writerow([path, label, f"{prob:.6f}"])
    print(f"Results saved to {results_path}")

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    auc_roc = roc_auc_score(all_labels, all_probs)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")

    metrics_path = output_dir / "wang2020_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accuracy", "precision", "recall", "auc_roc"])
        writer.writerow([f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{auc_roc:.4f}"])
    print(f"Metrics saved to {metrics_path}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
