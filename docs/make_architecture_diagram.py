"""Render the final-system architecture diagram (300 DPI PNG) for the report and
poster. Reflects the deployed design: FT-UnivFD forensics (no two-model gate) +
untargeted-PGD identity cloaking + JPEG Q=80, served by FastAPI and React.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[1]

INK = "#15171c"
INDIGO = "#4f46e5"
INDIGO_SOFT = "#eef0fe"
TEAL = "#0e8f8c"
TEAL_SOFT = "#e6f5f4"
GREEN = "#15803d"
GREEN_SOFT = "#e8f6ed"
RED = "#d63a3a"
SLATE = "#41454f"
GREYBOX = "#f3f3ef"


def box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=12.5, weight="normal", rounding=2.2):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"round,pad=0.4,rounding_size={rounding}",
                       linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            weight=weight, zorder=4, linespacing=1.35)


def arrow(ax, p1, p2, color=SLATE, lw=2.0, ls="-", style="-|>"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                        linewidth=lw, color=color, linestyle=ls,
                        shrinkA=3, shrinkB=3, zorder=2)
    ax.add_patch(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs/figures/architecture.png")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(12, 9.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # title
    ax.text(50, 97.5, "Hybrid Deepfake Defense System", ha="center", va="center",
            fontsize=21, weight="bold", color=INK)
    ax.text(50, 93.2, "Dual-layer architecture: forensic detection + adversarial identity cloaking",
            ha="center", va="center", fontsize=12.5, color=SLATE)

    # grouping panels
    ax.add_patch(FancyBboxPatch((4, 30), 44, 50, boxstyle="round,pad=0.4,rounding_size=2",
                                linewidth=0, facecolor=INDIGO_SOFT, alpha=0.5, zorder=0))
    ax.add_patch(FancyBboxPatch((52, 30), 44, 50, boxstyle="round,pad=0.4,rounding_size=2",
                                linewidth=0, facecolor=TEAL_SOFT, alpha=0.6, zorder=0))
    ax.text(26, 77.5, "FORENSICS LAYER", ha="center", fontsize=12, weight="bold", color=INDIGO)
    ax.text(26, 74.3, "detect synthetic media", ha="center", fontsize=10.5, color=SLATE)
    ax.text(74, 77.5, "PRIVACY LAYER", ha="center", fontsize=12, weight="bold", color=TEAL)
    ax.text(74, 74.3, "cloak a real identity", ha="center", fontsize=10.5, color=SLATE)

    # shared top
    box(ax, 50, 88, 34, 6.5, "Input image  ·  single still frame (PNG / JPEG)",
        "#ffffff", SLATE, fs=12.5)
    box(ax, 50, 70, 30, 6.5, "MTCNN face detection & crop", "#ffffff", SLATE, fs=12.5)
    arrow(ax, (50, 84.6), (50, 73.4))

    # forensics column (x=26)
    box(ax, 26, 60, 36, 7.5, "CLIP ViT-L/14\nfrozen backbone (open_clip, OpenAI)",
        "#ffffff", INDIGO, fs=12)
    box(ax, 26, 48, 36, 7.5, "Adversarial-aware probe\nLogisticRegression on raw features",
        "#ffffff", INDIGO, fs=12)
    box(ax, 26, 36, 36, 7.5, "Verdict: AUTHENTIC / SYNTHETIC\nthreshold 0.30  (no two-model gate)",
        GREEN_SOFT, GREEN, tc=GREEN, fs=12, weight="bold")
    arrow(ax, (26, 56.0), (26, 51.9), color=INDIGO)
    arrow(ax, (26, 44.0), (26, 39.9), color=INDIGO)

    # privacy column (x=74)
    box(ax, 74, 60, 36, 7.5, "Untargeted PGD\nε = 0.03, 40 steps, L∞", "#ffffff", TEAL, fs=12)
    box(ax, 74, 48, 36, 7.5, "Attack ArcFace\n(+ optional FaceNet)", "#ffffff", TEAL, fs=12)
    box(ax, 74, 36, 36, 7.5, "JPEG Q=80 → cloaked image\nidentity broken, looks unchanged",
        "#ffffff", TEAL, fs=12)
    arrow(ax, (74, 56.0), (74, 51.9), color=TEAL)
    arrow(ax, (74, 44.0), (74, 39.9), color=TEAL)

    # split from MTCNN to both columns
    arrow(ax, (44, 69), (30, 64.0), color=SLATE)
    arrow(ax, (56, 69), (70, 64.0), color=SLATE)

    # cross-link: cloaked image re-checked by the forensic probe
    arrow(ax, (56, 36), (44, 47), color=RED, lw=1.7, ls=(0, (5, 3)))
    ax.text(50, 43.0, "re-checked\n(cloak must read\nas authentic)", ha="center", va="center",
            fontsize=9.5, color=RED, linespacing=1.25)

    # serving layer
    box(ax, 50, 21, 78, 8.5,
        "FastAPI backend  (:8000, systemd)        React frontend  (nginx :80)\n"
        "/health     ·     /detect     ·     /protect",
        "#ffffff", SLATE, fs=12.5)
    arrow(ax, (26, 32.2), (35, 25.3), color=SLATE)
    arrow(ax, (74, 32.2), (65, 25.3), color=SLATE)

    box(ax, 50, 8.5, 46, 5.5, "Live demo  ·  http://34.135.192.253",
        INDIGO, INDIGO, tc="#ffffff", fs=12.5, weight="bold", rounding=1.6)
    arrow(ax, (50, 16.7), (50, 11.3), color=SLATE)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
