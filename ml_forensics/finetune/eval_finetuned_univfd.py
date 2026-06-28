import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned UnivFD linear probe.")
    parser.add_argument("--probe",         required=True, help="Path to .pkl probe file")
    parser.add_argument("--features",      required=True, help="Path to .npz file with features/labels/paths")
    parser.add_argument("--output_prefix", required=True, help="Output path prefix (e.g. results/phase2/forensics/ft_univfd_ff)")
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    clf = joblib.load(args.probe)
    data = np.load(args.features, allow_pickle=True)
    X, y_true, paths = data["features"], data["labels"], data["paths"]
    print(f"Loaded {len(X)} samples  ({(y_true == 0).sum()} real, {(y_true == 1).sum()} fake)")

    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, y_prob),
    }

    pd.DataFrame([metrics]).to_csv(f"{prefix}_metrics.csv", index=False)
    pd.DataFrame({
        "image_path":     paths,
        "true_label":     y_true,
        "predicted_prob": y_prob,
    }).to_csv(f"{prefix}_results.csv", index=False)

    print(f"Results  → {prefix}_results.csv")
    print(f"Metrics  → {prefix}_metrics.csv")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
