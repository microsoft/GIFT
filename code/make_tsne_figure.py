"""t-SNE figure for the random-negative misclassification analysis (reads pairs_random.csv with tsne1/tsne2)."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

plt.rcParams.update({"font.size": 8.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})
STYLES = [  # outcome, label, color, marker, size (full map), zorder
    ("TN", "True negative", "#cfcfcf", "o", 3, 1),
    ("TP", "True positive", "#2f6db5", "o", 4, 2),
    ("FP", "False positive", "#f08c00", "^", 12, 4),
    ("FN", "False negative", "#c62828", "D", 7, 3),
]


def draw(ax, df, scale):
    for k, lab, col, mk, sz, z in STYLES:
        m = df.outcome == k
        err = k in ("FP", "FN")
        ax.scatter(df.tsne1[m], df.tsne2[m], s=sz * scale, c=col, marker=mk, alpha=1 if err else 0.75,
                   linewidths=0.3 if err else 0, edgecolors="white" if err else "none", zorder=z,
                   label=f"{lab} (n={m.sum():,})", rasterized=not err)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/gift/run_42/misclassification/pairs_random.csv")
    ap.add_argument("--out", default="manuscript/figs/tsne_pairs_predictions")
    ap.add_argument("--left_label", default="compounds without\ntraining interactions")
    ap.add_argument("--box_label", default="compounds with\ntraining interactions")
    ap.add_argument("--box_xmin_frac", type=float, default=None,
                    help="box all pairs right of this fraction of the t-SNE 1 range instead of the positives")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.12})
    draw(a1, df, 1.0)
    a1.set_xlabel("t-SNE 1", fontsize=8); a1.set_ylabel("t-SNE 2", fontsize=8)
    a1.set_title("(a) All test pairs", fontsize=9, loc="left")

    # Zoom on the group of pairs whose compound has training interactions (contains the positives).
    pos = df[df.y == 1]
    if args.box_xmin_frac is not None:
        pos = df[df.tsne1 >= df.tsne1.min() + args.box_xmin_frac * (df.tsne1.max() - df.tsne1.min())]
    x0, x1 = pos.tsne1.quantile(0.01), pos.tsne1.quantile(0.99)
    y0, y1 = pos.tsne2.quantile(0.01), pos.tsne2.quantile(0.99)
    pad = 0.12 if args.box_xmin_frac is None else 0.04
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    a1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.7, ls="--", ec="#444444", zorder=5))
    a1.text(x0 + (x1 - x0) / 2, y1, "(b)", ha="center", va="bottom", fontsize=7.5, color="#444444")
    xmid = (df.tsne1.min() + x0) / 2
    a1.text(xmid, df.tsne2.max(), args.left_label, ha="center", va="bottom",
            fontsize=7, color="#666666")
    a1.text((x0 + x1) / 2, y1 + 0.09 * (df.tsne2.max() - df.tsne2.min()), args.box_label,
            ha="center", va="bottom", fontsize=7, color="#666666")
    ylo, yhi = a1.get_ylim()
    a1.set_ylim(ylo, yhi + 0.1 * (yhi - ylo))

    draw(a2, df, 1.8)
    a2.set_xlim(x0, x1); a2.set_ylim(y0, y1)
    for side in a2.spines.values():
        side.set_visible(True); side.set_linestyle("--"); side.set_color("#444444"); side.set_linewidth(0.7)
    a2.set_title("(b) Boxed region, enlarged", fontsize=9, loc="left")

    h, l = a1.get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False,
                     handletextpad=0.2, columnspacing=1.4, markerscale=1.8)
    for hh in leg.legend_handles:
        hh.set_alpha(1)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=400)


if __name__ == "__main__":
    main()
