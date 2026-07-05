"""Generate report/poster figures from the Phase 2 result CSVs.

All figures are rendered at 300 DPI with a consistent editorial palette and saved
to results/phase2/figures/. A curated subset is also copied to docs/figures/ for
the final report. Plots are generated from CSVs only (per CLAUDE.md).
"""

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

REPO_ROOT = Path(__file__).resolve().parents[2]

# palette (matches the web app)
INDIGO = "#4f46e5"
TEAL = "#0ea5a4"
RED = "#d63a3a"
AMBER = "#d97706"
GREEN = "#15803d"
SLATE = "#41454f"
FAINT = "#9aa0ab"
GRID = "#e8e8e2"


def set_style():
    # Prefer Inter / Helvetica if present, else fall back cleanly.
    avail = {f.name for f in fm.fontManager.ttflist}
    family = next((f for f in ["Inter", "Helvetica", "Arial", "DejaVu Sans"] if f in avail), "sans-serif")
    plt.rcParams.update({
        "font.family": family,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.edgecolor": SLATE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "text.color": "#15171c",
        "axes.labelcolor": "#15171c",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "figure.dpi": 300,
    })


def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(REPO_ROOT)}")
    return p


def bar_labels(ax, bars, fmt="{:.2f}", dy=0.012, fontsize=10):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, color=SLATE)


# ----------------------------------------------------------------------
def fig_conflict_curve(conflict_dir, out_dir):
    rows = read_csv_rows(conflict_dir / "cloaked_fpr_curve.csv")
    t = [float(r["threshold"]) for r in rows]
    overall = [float(r["cloaked_fpr_overall"]) for r in rows]
    arc = [float(r["cloaked_fpr_arcface"]) for r in rows]
    fnet = [float(r["cloaked_fpr_facenet"]) for r in rows]
    recall = [float(r["ff_recall"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(t, overall, "-o", color=INDIGO, lw=2.4, ms=5, label="Cloaked FPR (all 400)")
    ax.plot(t, arc, "--o", color=TEAL, lw=1.8, ms=4, label="Cloaked FPR (ArcFace)")
    ax.plot(t, fnet, "--o", color=RED, lw=1.8, ms=4, label="Cloaked FPR (FaceNet)")
    ax.plot(t, recall, "-s", color=FAINT, lw=1.8, ms=4, label="FF++ recall")
    ax.axhline(0.15, color=AMBER, ls=":", lw=1.6)
    ax.text(0.885, 0.165, "15% target", color=AMBER, fontsize=10, va="bottom", ha="right")
    ax.axvline(0.30, color="#15171c", lw=1.0, alpha=0.35)
    ax.scatter([0.30], [0.1775], color=INDIGO, zorder=5, s=70, edgecolor="white", linewidth=1.5)
    ax.annotate("operating point\nt=0.30  (FPR 0.177, recall 0.62)",
                xy=(0.30, 0.1775), xytext=(0.42, 0.40),
                fontsize=10, color="#15171c",
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.0))
    ax.set_xlabel("Detection threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Forensic conflict: cloaked false-positive rate vs threshold")
    ax.set_ylim(-0.02, 0.82)
    ax.legend(loc="upper right", fontsize=10)
    return save(fig, out_dir, "fig_conflict_curve.png")


def fig_mitigation_stages(out_dir):
    labels = ["Naive probe\n(t=0.50)", "Adv-aware probe\n(t=0.30)",
              "Adv-aware + JPEG Q80\n(ArcFace, t=0.30)"]
    vals = [0.895, 0.1775, 0.02]
    colors = [RED, AMBER, GREEN]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.axhline(0.15, color=SLATE, ls=":", lw=1.4)
    ax.text(2.45, 0.16, "15% target", color=SLATE, fontsize=10, va="bottom", ha="right")
    bar_labels(ax, bars, fmt="{:.1%}", dy=0.015, fontsize=12)
    ax.set_ylabel("Cloaked-image false-positive rate")
    ax.set_title("Conflict mitigation: cloaked FPR across stages")
    ax.set_ylim(0, 1.0)
    return save(fig, out_dir, "fig_mitigation_stages.png")


def fig_forensic_models(out_dir):
    fdir = REPO_ROOT / "results/forensics"

    def auc(path):
        return float(read_csv_rows(path)[0]["auc_roc"])

    models = [
        ("Wang2020", "wang2020"),
        ("Gragnaniello2021", "gragnaniello2021"),
        ("Mandelli2022", "mandelli2022"),
        ("UnivFD (baseline)", "univfd"),
    ]
    ff, pg, names = [], [], []
    for disp, key in models:
        ff_p = fdir / f"ff_plus_plus/{key}_ff_metrics.csv"
        pg_p = fdir / f"progan/{key}_progan_metrics.csv"
        if key == "univfd":
            pg_p = fdir / "progan/univfd_metrics.csv"
        if ff_p.exists() and pg_p.exists():
            names.append(disp)
            ff.append(auc(ff_p))
            pg.append(auc(pg_p))
    # deployed model (from phase2 corrected)
    names.append("FT-UnivFD (ours)")
    ff.append(0.812)
    pg.append(0.829)

    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    b1 = ax.bar([i - w / 2 for i in x], ff, w, label="FF++ AUC", color=INDIGO)
    b2 = ax.bar([i + w / 2 for i in x], pg, w, label="ProGAN AUC", color=TEAL)
    # highlight ours
    b1[-1].set_color("#312e81")
    b2[-1].set_color("#0b7d7b")
    bar_labels(ax, b1, fmt="{:.2f}", fontsize=9)
    bar_labels(ax, b2, fmt="{:.2f}", fontsize=9)
    ax.axhline(0.5, color=FAINT, ls=":", lw=1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0, 1.12)
    ax.set_title("Forensic detectors: FF++ vs ProGAN AUC")
    ax.legend(loc="upper left", fontsize=10)
    return save(fig, out_dir, "fig_forensic_models.png")


def fig_privacy_quality(out_dir):
    rows = read_csv_rows(REPO_ROOT / "results/phase2/robustness/robustness_summary.csv")
    base = next(r for r in rows if r["transform"].startswith("Baseline"))
    arc = [float(base["arcface_asr"]), float(base["arcface_ssim"]), float(base["arcface_psnr"])]
    fnet = [float(base["facenet_asr"]), float(base["facenet_ssim"]), float(base["facenet_psnr"])]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    titles = ["Attack Success Rate", "SSIM", "PSNR (dB)"]
    ylims = [(0, 1.15), (0, 1.05), (0, 40)]
    for i, (ax, title, yl) in enumerate(zip(axes, titles, ylims)):
        bars = ax.bar(["ArcFace", "FaceNet"], [arc[i], fnet[i]], color=[INDIGO, TEAL], width=0.55)
        fmt = "{:.3f}" if i == 1 else ("{:.2f}" if i == 0 else "{:.1f}")
        bar_labels(ax, bars, fmt=fmt, dy=yl[1] * 0.01, fontsize=11)
        ax.set_title(title)
        ax.set_ylim(*yl)
    fig.suptitle("Identity cloaking quality (untargeted PGD, ε=0.03, 40 steps)",
                 fontsize=14, fontweight="bold", y=1.02)
    return save(fig, out_dir, "fig_privacy_quality.png")


def fig_robustness(out_dir):
    rows = read_csv_rows(REPO_ROOT / "results/phase2/robustness/robustness_summary.csv")
    names = [r["transform"] for r in rows]
    arc = [float(r["arcface_asr"]) for r in rows]
    fnet = [float(r["facenet_asr"]) for r in rows]
    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    b1 = ax.bar([i - w / 2 for i in x], arc, w, label="ArcFace", color=INDIGO)
    b2 = ax.bar([i + w / 2 for i in x], fnet, w, label="FaceNet", color=TEAL)
    bar_labels(ax, b1, fmt="{:.2f}", fontsize=9)
    bar_labels(ax, b2, fmt="{:.2f}", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=10)
    ax.set_ylabel("Attack Success Rate (ASR)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Cloaking robustness under image transforms")
    ax.legend(loc="center right", fontsize=10)
    return save(fig, out_dir, "fig_robustness.png")


def fig_jpeg_sweep(out_dir):
    rows = read_csv_rows(REPO_ROOT / "results/phase2/privacy/exp2_jpeg_sweep.csv")
    rows = sorted(rows, key=lambda r: int(r["Q"]))
    q = [int(r["Q"]) for r in rows]
    asr = [float(r["arcface_asr"]) for r in rows]
    fpr = [float(r["forensic_cloaked_fpr@0.30"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(q, asr, "-o", color=INDIGO, lw=2.2, ms=5, label="ArcFace ASR")
    ax.plot(q, fpr, "-s", color=RED, lw=2.2, ms=5, label="Forensic cloaked FPR @0.30")
    ax.axvline(80, color="#15171c", lw=1.0, alpha=0.35)
    ax.annotate("chosen: Q=80\nASR 0.99, FPR 0.02", xy=(80, 0.5), xytext=(62, 0.62),
                fontsize=10, color="#15171c",
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.0))
    ax.set_xlabel("JPEG quality applied to cloaked crop")
    ax.set_ylabel("Rate")
    ax.set_title("Post-cloak JPEG: attack success vs forensic detectability")
    ax.set_ylim(-0.03, 1.08)
    ax.invert_xaxis()  # stronger compression (lower Q) to the right
    ax.legend(loc="center left", fontsize=10)
    return save(fig, out_dir, "fig_jpeg_sweep.png")


def fig_epsilon_tradeoff(out_dir):
    pdir = REPO_ROOT / "results/phase2/privacy"
    pts = []
    for p in sorted(pdir.glob("pgd_untargeted_arcface_eps*_steps40_metrics.csv")):
        r = read_csv_rows(p)[0]
        pts.append((float(r["epsilon"]), float(r["mean_asr"]),
                    float(r["mean_ssim"]), float(r["mean_psnr"])))
    pts.sort()
    eps = [p[0] for p in pts]
    asr = [p[1] for p in pts]
    ssim = [p[2] for p in pts]
    psnr = [p[3] for p in pts]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(eps, asr, "-o", color=INDIGO, lw=2.2, ms=5, label="ArcFace ASR")
    ax.plot(eps, ssim, "-s", color=TEAL, lw=2.2, ms=5, label="SSIM")
    ax.set_xlabel("PGD perturbation budget ε (L∞)")
    ax.set_ylabel("ASR / SSIM")
    ax.set_ylim(0, 1.08)
    ax2 = ax.twinx()
    ax2.plot(eps, psnr, "-^", color=AMBER, lw=1.8, ms=5, label="PSNR (dB)")
    ax2.set_ylabel("PSNR (dB)", color=AMBER)
    ax2.tick_params(axis="y", colors=AMBER)
    ax2.grid(False)
    ax.axvline(0.03, color="#15171c", lw=1.0, alpha=0.35)
    ax.text(0.031, 0.06, "chosen ε=0.03", fontsize=10, color="#15171c")
    ax.set_title("Cloaking trade-off vs perturbation budget (ArcFace)")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="center right", fontsize=10)
    return save(fig, out_dir, "fig_epsilon_tradeoff.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=REPO_ROOT / "results/phase2/figures")
    ap.add_argument("--docs_dir", type=Path, default=REPO_ROOT / "docs/figures/results")
    args = ap.parse_args()

    set_style()
    conflict_dir = REPO_ROOT / "results/phase2/conflict"
    print("Generating figures...")
    made = [
        fig_conflict_curve(conflict_dir, args.out_dir),
        fig_mitigation_stages(args.out_dir),
        fig_forensic_models(args.out_dir),
        fig_privacy_quality(args.out_dir),
        fig_robustness(args.out_dir),
        fig_jpeg_sweep(args.out_dir),
        fig_epsilon_tradeoff(args.out_dir),
    ]
    # copy all to docs/figures/results for the report
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    for p in made:
        shutil.copy(p, args.docs_dir / p.name)
    print(f"\nCopied {len(made)} figures to {args.docs_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
