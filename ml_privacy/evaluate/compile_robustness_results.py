"""
Compile robustness evaluation results into a summary table.

Loads 6 per-image ASR CSVs from results/phase2/robustness/ and combines
them with Phase 2 baselines to produce a formatted table and summary CSV.

Usage:
  python compile_robustness_results.py
  python compile_robustness_results.py --results_dir results/phase2/robustness
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Phase 2 baseline (no transform) — from PGD eps=0.03 steps=40 run
BASELINES = {
    "arcface": {"asr": 1.000, "ssim": 0.902, "psnr": 34.21},
    "facenet": {"asr": 1.000, "ssim": 0.863, "psnr": 33.54},
}

TRANSFORMS = [
    ("jpeg75",     "JPEG Q=75"),
    ("jpeg50",     "JPEG Q=50"),
    ("downsample", "Bilinear DS/US"),
]

RECOGNIZERS = ["arcface", "facenet"]


def load_asr_csv(path):
    """Return (mean_asr, mean_ssim, mean_psnr) from a per-image ASR CSV."""
    if not path.exists():
        print(f"  WARN: file not found: {path}")
        return float("nan"), float("nan"), float("nan")

    df = pd.read_csv(path)

    mean_asr = float(df["asr"].mean())

    ssim_vals = pd.to_numeric(df["ssim"], errors="coerce").dropna()
    psnr_vals = pd.to_numeric(df["psnr"], errors="coerce").dropna()
    mean_ssim = float(ssim_vals.mean()) if len(ssim_vals) else float("nan")
    mean_psnr = float(psnr_vals.mean()) if len(psnr_vals) else float("nan")

    return mean_asr, mean_ssim, mean_psnr


def fmt(val, decimals=3):
    if val != val:  # nan check
        return "N/A"
    return f"{val:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compile robustness summary table from 6 ASR CSVs"
    )
    parser.add_argument(
        "--results_dir", type=Path,
        default=REPO_ROOT / "results/phase2/robustness",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    print(f"Results dir: {results_dir}\n")

    # Build table rows
    table_rows = []

    # Baseline row
    table_rows.append({
        "transform":       "Baseline (no tx)",
        "arcface_asr":     BASELINES["arcface"]["asr"],
        "arcface_ssim":    BASELINES["arcface"]["ssim"],
        "arcface_psnr":    BASELINES["arcface"]["psnr"],
        "facenet_asr":     BASELINES["facenet"]["asr"],
        "facenet_ssim":    BASELINES["facenet"]["ssim"],
        "facenet_psnr":    BASELINES["facenet"]["psnr"],
    })

    for tx_key, tx_label in TRANSFORMS:
        row = {"transform": tx_label}
        for rec in RECOGNIZERS:
            csv_path = results_dir / f"{tx_key}_{rec}_asr.csv"
            asr, ssim, psnr = load_asr_csv(csv_path)
            row[f"{rec}_asr"]  = asr
            row[f"{rec}_ssim"] = ssim
            row[f"{rec}_psnr"] = psnr
        table_rows.append(row)

    # Print table
    header = (
        f"{'Transform':<20} | {'ArcFace ASR':>11} | {'ArcFace SSIM':>12} | "
        f"{'ArcFace PSNR':>12} | {'FaceNet ASR':>11} | {'FaceNet SSIM':>12} | "
        f"{'FaceNet PSNR':>12}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in table_rows:
        print(
            f"{r['transform']:<20} | "
            f"{fmt(r['arcface_asr']):>11} | "
            f"{fmt(r['arcface_ssim']):>12} | "
            f"{fmt(r['arcface_psnr'], 2):>12} | "
            f"{fmt(r['facenet_asr']):>11} | "
            f"{fmt(r['facenet_ssim']):>12} | "
            f"{fmt(r['facenet_psnr'], 2):>12}"
        )
    print(sep)

    # Save CSV
    df_out = pd.DataFrame(table_rows)
    out_path = results_dir / "robustness_summary.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
