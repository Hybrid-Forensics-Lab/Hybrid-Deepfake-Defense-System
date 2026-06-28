import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Summarise ArcFace PGD pilot sweep results.")
    p.add_argument("--input_dir", type=Path, default=Path("results/phase2/privacy"))
    p.add_argument("--output", type=Path,
                   default=Path("results/phase2/privacy/pilot_summary.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(args.input_dir.glob("pgd_untargeted_arcface_eps*_steps*_metrics.csv"))

    if not files:
        print(f"No metrics CSVs found in {args.input_dir} — run the pilot sweep first.")
        return

    rows = [pd.read_csv(f).iloc[0].to_dict() for f in files]
    df = pd.DataFrame(rows).sort_values(["epsilon", "steps"]).reset_index(drop=True)

    print("\nFull pilot results table:")
    print(df.to_string(index=False))

    for metric, label in [("mean_asr", "ASR"), ("mean_ssim", "SSIM"), ("mean_psnr", "PSNR (dB)")]:
        pivot = df.pivot_table(index="epsilon", columns="steps", values=metric, aggfunc="first")
        print(f"\n{label} (epsilon × steps):")
        print(pivot.to_string())

    # Operating point selection: highest epsilon where SSIM >= 0.85 AND PSNR >= 30 AND meaningful ASR
    candidates = df[(df["mean_ssim"] >= 0.85) & (df["mean_psnr"] >= 30.0)]
    if not candidates.empty:
        best = candidates.loc[candidates["mean_asr"].idxmax()]
        print(f"\nRecommended operating point:")
        print(f"  epsilon={best['epsilon']}  steps={int(best['steps'])}"
              f"  ASR={best['mean_asr']:.4f}  SSIM={best['mean_ssim']:.4f}"
              f"  PSNR={best['mean_psnr']:.2f} dB")
    else:
        best = df.loc[df["mean_asr"].idxmax()]
        print(f"\nNo point meets SSIM>=0.85 and PSNR>=30 — best ASR point:")
        print(f"  epsilon={best['epsilon']}  steps={int(best['steps'])}"
              f"  ASR={best['mean_asr']:.4f}  SSIM={best['mean_ssim']:.4f}"
              f"  PSNR={best['mean_psnr']:.2f} dB  (quality trade-off — document in Ch. 5.3)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
