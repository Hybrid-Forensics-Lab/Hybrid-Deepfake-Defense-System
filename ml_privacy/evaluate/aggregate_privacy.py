import os
import csv
from pathlib import Path

# Adjust this to wherever your privacy CSVs are
RESULTS_DIR = Path("results/privacy")

# Files expected: fawkes_arcface_metrics.csv, fawkes_facenet_metrics.csv,
#                 pgd_untargeted_arcface_metrics.csv, pgd_untargeted_facenet_metrics.csv,
#                 pgd_targeted_arcface_metrics.csv, pgd_targeted_facenet_metrics.csv

files = [
    "fawkes_arcface_metrics.csv",
    "fawkes_facenet_metrics.csv",
    "pgd_untargeted_arcface_metrics.csv",
    "pgd_untargeted_facenet_metrics.csv",
    "pgd_targeted_arcface_metrics.csv",
    "pgd_targeted_facenet_metrics.csv",
]

print(f"{'File':<45} {'N':>5} {'ASR':>7} {'SSIM':>7} {'PSNR':>7}")
print("-" * 75)

for fname in files:
    fpath = RESULTS_DIR / fname
    if not fpath.exists():
        print(f"{fname:<45} NOT FOUND")
        continue

    rows = []
    with open(fpath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"{fname:<45} EMPTY")
        continue

    n = len(rows)
    # ASR: proportion where correct == 0 (recognition failed = attack succeeded)
    # Handle both int and string representations
    asr = sum(1 for r in rows if str(r.get("correct", "1")).strip() in ("0", "False", "false")) / n
    mean_ssim = sum(float(r["ssim"]) for r in rows) / n
    mean_psnr = sum(float(r["psnr"]) for r in rows) / n

    print(f"{fname:<45} {n:>5} {asr:>7.4f} {mean_ssim:>7.4f} {mean_psnr:>7.2f}")
