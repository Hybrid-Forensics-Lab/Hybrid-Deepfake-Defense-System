"""
Compile all FPR CSVs from results/phase2/conflict/ into a single summary table.
Also pulls FF++ recall for each model from Phase 2 forensics results.
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

FF_RESULTS_PATHS = {
    "wang2020":   REPO_ROOT / "results/forensics/wang2020_ff_results.csv",
    "univfd_knn": REPO_ROOT / "results/forensics/univfd_ff_results.csv",
    "ft_univfd":  REPO_ROOT / "results/phase2/forensics/ft_univfd_mlp_ff_results.csv",
}

FF_RECALL_FALLBACK = {
    "wang2020":   0.010,   # Phase 1: Wang2020 FF++ recall ~1% (AUC 0.530, near-random)
    "univfd_knn": None,
    "ft_univfd":  None,
}

_ff_results_cache = {}


def load_ff_recall_at_threshold(model_name, threshold):
    if model_name not in _ff_results_cache:
        src = FF_RESULTS_PATHS.get(model_name)
        if src and src.exists():
            _ff_results_cache[model_name] = pd.read_csv(src)
        else:
            _ff_results_cache[model_name] = None

    df = _ff_results_cache.get(model_name)
    if df is None:
        return FF_RECALL_FALLBACK.get(model_name)

    from sklearn.metrics import recall_score
    preds = (df["predicted_prob"] >= threshold).astype(int)
    return recall_score(df["true_label"], preds, zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conflict_dir", default=None,
                        help="Directory with FPR CSVs (default: results/phase2/conflict/)")
    parser.add_argument("--output", default=None,
                        help="Output summary CSV (default: results/phase2/conflict/conflict_summary.csv)")
    args = parser.parse_args()

    conflict_dir = Path(args.conflict_dir) if args.conflict_dir else REPO_ROOT / "results/phase2/conflict"
    output_path  = Path(args.output) if args.output else conflict_dir / "conflict_summary.csv"

    # Collect all per-run metrics CSVs (exclude per_image and summary files)
    csv_files = sorted(
        p for p in conflict_dir.glob("*_fpr.csv")
        if "per_image" not in p.name and "summary" not in p.name
    )

    if not csv_files:
        print(f"No *_fpr.csv files found in {conflict_dir}")
        return

    rows = []
    for f in csv_files:
        df = pd.read_csv(f)
        rows.append(df.iloc[0].to_dict())

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["forensic_model", "threshold"]).reset_index(drop=True)

    # Add FF++ recall at each row's specific threshold
    summary["ff_recall"] = summary.apply(
        lambda r: load_ff_recall_at_threshold(r["forensic_model"], r["threshold"]), axis=1
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"\n{'Model':<16} {'Thresh':>7} {'FPR':>8} {'FF++ Recall':>12} {'<15%':>6}")
    print("-" * 54)
    for _, row in summary.iterrows():
        recall_str = f"{row['ff_recall']:.3f}" if pd.notna(row.get('ff_recall')) else "  N/A"
        passes = "YES" if row["passes_15pct"] else "no"
        print(f"{row['forensic_model']:<16} {row['threshold']:>7.1f} "
              f"{row['fpr']:>8.4f} {recall_str:>12} {passes:>6}")

    print(f"\nSummary saved → {output_path}")

    # Print operating point recommendation
    passing = summary[summary["passes_15pct"] == True]
    if not passing.empty:
        best = passing.sort_values("ff_recall", ascending=False).iloc[0]
        print(f"\nBest passing config: model={best['forensic_model']}, "
              f"threshold={best['threshold']}, FPR={best['fpr']:.4f}, "
              f"FF++ recall={best.get('ff_recall', 'N/A')}")
    else:
        print("\nNo configuration achieves FPR < 15% — two-model gate required.")


if __name__ == "__main__":
    main()
