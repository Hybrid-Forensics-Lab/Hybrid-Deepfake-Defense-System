import argparse
import sys
import torch
import torchvision.transforms as transforms
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


# CLIP normalization — UnivFD CLIP models require CLIP stats, not ImageNet
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def load_model(univfd_dir, weights_path, device):
    sys.path.insert(0, str(univfd_dir))
    from models import get_model

    model = get_model("CLIP:ViT-L/14")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.fc.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def run_inference(model, manifest_path, repo_root, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    df = pd.read_csv(manifest_path)
    rows = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="UnivFD inference"):
            img_path = repo_root / row["image_path"]
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            prob = model(tensor).sigmoid().item()
            rows.append({"image_path": row["image_path"], "true_label": int(row["label"]), "predicted_prob": prob})

    return pd.DataFrame(rows)


def compute_metrics(results_df):
    y_true = results_df["true_label"].values
    y_prob = results_df["predicted_prob"].values
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, y_prob),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",  default="data/progan_test/manifest.csv")
    parser.add_argument("--repo_root", default=None,
                        help="Repo root path. Defaults to two levels above this script.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[2]
    manifest_path = repo_root / args.manifest

    univfd_dir   = repo_root / "ml_forensics" / "models" / "UniversalFakeDetect"
    weights_path = univfd_dir / "pretrained_weights" / "fc_weights.pth"

    out_dir = repo_root / "results" / "forensics" / "progan"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "univfd_results.csv"
    metrics_csv = out_dir / "univfd_metrics.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(univfd_dir, weights_path, device)
    print("Model loaded.")

    results_df = run_inference(model, manifest_path, repo_root, device)
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")

    del model
    torch.cuda.empty_cache()

    metrics = compute_metrics(pd.read_csv(results_csv))
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    print(f"Metrics saved to {metrics_csv}")

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
