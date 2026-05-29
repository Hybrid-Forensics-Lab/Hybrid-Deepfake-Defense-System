import argparse
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def normalize_path(p_str, repo):
    p = Path(p_str)
    if p.is_absolute():
        return str(p.resolve())
    return str((repo / p).resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_csv', default='results/forensics/gragnaniello2021_raw_stylegan2.csv')
    parser.add_argument('--manifest', default='data/ff_plus_plus/256x256/manifest.csv')
    parser.add_argument('--results_csv', default='results/forensics/gragnaniello2021_results.csv')
    parser.add_argument('--metrics_csv', default='results/forensics/gragnaniello2021_metrics.csv')
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    raw_path = repo / args.raw_csv
    manifest_path = repo / args.manifest
    results_path = repo / args.results_csv
    metrics_path = repo / args.metrics_csv

    raw_df = pd.read_csv(raw_path)
    print("Raw CSV columns:", raw_df.columns.tolist())
    print("First 5 rows:")
    print(raw_df.head())
    print()

    manifest_df = pd.read_csv(manifest_path)

    scores_outside_unit = (raw_df['logit'] < 0) | (raw_df['logit'] > 1)
    apply_sigmoid = scores_outside_unit.any()
    if apply_sigmoid:
        print("Scores outside [0,1] detected — applying sigmoid.")
        raw_df['predicted_prob'] = raw_df['logit'].apply(sigmoid)
    else:
        print("Scores already in [0,1] — using as-is.")
        raw_df['predicted_prob'] = raw_df['logit']

    manifest_df['_norm_path'] = manifest_df['image_path'].apply(lambda p: normalize_path(p, repo))
    raw_df['_norm_path'] = raw_df['filename'].apply(lambda p: normalize_path(p, repo))

    merged = manifest_df.merge(raw_df[['_norm_path', 'predicted_prob']], on='_norm_path', how='inner')
    print(f"Merged rows: {len(merged)} / {len(manifest_df)} manifest rows")
    if len(merged) != len(manifest_df):
        print(f"WARNING: {len(manifest_df) - len(merged)} manifest rows had no match in raw CSV.")

    results_df = merged[['image_path', 'label', 'predicted_prob']].rename(columns={'label': 'true_label'})
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    y_true = results_df['true_label'].values
    y_prob = results_df['predicted_prob'].values
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    metrics_df = pd.DataFrame([{'accuracy': acc, 'precision': prec, 'recall': rec, 'auc_roc': auc}])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    print(f"\nRow count: {len(results_df)}")


if __name__ == '__main__':
    main()
