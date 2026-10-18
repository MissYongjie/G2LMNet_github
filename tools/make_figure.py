"""Compose the qualitative comparison figure (paper Fig. "error maps") from figures/comparison/.

Example:
    python tools/make_figure.py --out figures/qualitative_comparison.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

COLUMNS = [
    ("t1.png", "(a) $X_1$"),
    ("t2.png", "(b) $X_2$"),
    ("gt.png", "(c) GT"),
    ("IRG-McS_error.png", "(d) IRG-McS"),
    ("NPSG_error.png", "(e) NPSG"),
    ("SCASC_error.png", "(f) SCASC"),
    ("GIR-MRF_error.png", "(g) GIR-MRF"),
    ("SDIR_error.png", "(h) SDIR"),
    ("RIEM-O_error.png", "(i) RIEM-O"),
    ("RIEM-L_error.png", "(j) RIEM-L"),
    ("G2LMNet_error.png", "(k) G$^2$LMNet"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="figures/comparison")
    parser.add_argument("--out", default="figures/qualitative_comparison.png")
    parser.add_argument("--cell_width", type=float, default=1.25, help="inches per column")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    rows = [f"hetdata{i}" for i in range(1, 7)]
    ratios = []
    for r in rows:
        w, h = Image.open(os.path.join(args.src, r, "t1.png")).size
        ratios.append(h / w)
    fig_w = args.cell_width * (len(COLUMNS) + 0.35)
    fig_h = args.cell_width * (sum(ratios) + 0.45)
    fig, axes = plt.subplots(len(rows), len(COLUMNS), figsize=(fig_w, fig_h),
                             gridspec_kw={"height_ratios": ratios, "wspace": 0.04, "hspace": 0.04})
    for i, r in enumerate(rows):
        for j, (name, title) in enumerate(COLUMNS):
            ax = axes[i, j]
            img = Image.open(os.path.join(args.src, r, name))
            ax.imshow(img, cmap="gray" if img.mode == "L" else None, interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"#{i + 1}", fontsize=9, rotation=0, labelpad=12, va="center")
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
