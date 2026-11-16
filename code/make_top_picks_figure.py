"""Top-picks figure for the lead prioritization: model rank of the top-ranked drugs at each driver kinase,
with the ChEMBL evidence for every drug-driver pair. Optional second run (e.g. the no-soft-edge ablation)
adds a panel comparing the known actives among the 10 top-ranked drugs."""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "legend.fontsize": 8.5, "axes.linewidth": 0.8, "pdf.fonttype": 42})
EVIDENCE = {"active": ("Known active (ChEMBL 36)", "#74c476"),
            "active_parent": ("Active, parent compound (ChEMBL 37)", "#c7e9c0"),
            "tested": ("Tested, no pChEMBL $\\geq$ 5", "#fcbba1"),
            "untested": ("No activity record", "#f0f0f0")}
SALT = re.compile(r"\s+(ANHYDROUS|DIMALEATE|HYDROCHLORIDE|ISETHIONATE|SULFATE|ESYLATE|MESYLATE)$")
DRIVER_ORDER = ["EGFR", "ERBB2", "ALK", "ROS1", "MET", "RET", "BRAF", "NTRK1", "NTRK2", "NTRK3"]


def clean(name):
    return SALT.sub("", name).capitalize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="results/gift/run_42")
    parser.add_argument("--ablation_dir", default="")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--names", default="data/chembl_names.json")
    parser.add_argument("--web", default="data/chembl37_driver_activities.json")
    parser.add_argument("--out", default="figures/lead_top_picks")
    args = parser.parse_args()

    d = json.load(open(os.path.join(args.run_dir, "lead_prioritization.json")))
    names = json.load(open(args.names))["compounds"]
    web = json.load(open(args.web))
    gene_to_t = {g: t for t, g in d["driver_targets"].items()}
    targets = [gene_to_t[g] for g in DRIVER_ORDER if g in gene_to_t]
    scores = d["driver_scores"]
    ranks = {t: {c: r + 1 for r, c in enumerate(sorted(scores[t], key=lambda c: -scores[t][c]))} for t in targets}

    picks = []
    for t in targets:
        for c in sorted(scores[t], key=lambda c: -scores[t][c])[: args.top]:
            if c not in picks:
                picks.append(c)
    picks.sort(key=lambda c: np.mean([ranks[t][c] for t in targets]))

    def evidence(c, t):
        if c in d["driver_labels"][t]:
            return "active"
        rec = web.get(c, {}).get("targets", {}).get(t)
        if rec and (rec["max_pchembl"] or 0) >= 5:
            return "active_parent"
        return "tested" if rec else "untested"

    has_abl = bool(args.ablation_dir)
    fig = plt.figure(figsize=(7.2, 0.24 * len(picks) + 1.4))
    gs = fig.add_gridspec(1, 2 if has_abl else 1, width_ratios=[1.8, 1] if has_abl else [1], wspace=0.3)
    ax = fig.add_subplot(gs[0])

    # (a) one cell per drug-driver pair: color = recorded evidence, number = model rank (top ranks in bold).
    for i, c in enumerate(picks):
        for j, t in enumerate(targets):
            r = ranks[t][c]
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=EVIDENCE[evidence(c, t)][1],
                                       edgecolor="white", lw=1.5))
            top = r <= args.top
            ax.text(j, i, str(r), ha="center", va="center", fontsize=8 if top else 6.5,
                    fontweight="bold" if top else "normal", color="black" if top else "#404040")
    ax.set_xlim(-0.5, len(targets) - 0.5)
    ax.set_ylim(len(picks) - 0.5, -0.5)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(targets)), [d["driver_targets"][t] for t in targets], rotation=90, fontsize=8)
    ax.set_yticks(range(len(picks)), [clean(names[c]["name"]) for c in picks])
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=col, edgecolor="none", label=lab)
               for lab, col in EVIDENCE.values()]

    ax2 = None
    if has_abl:
        a = json.load(open(os.path.join(args.ablation_dir, "lead_prioritization.json")))
        ax2 = fig.add_subplot(gs[1])
        y = np.arange(len(targets))
        full = [d["per_target"][t]["gnn"]["hits_top"] for t in targets]
        abl = [a["per_target"][t]["gnn"]["hits_top"] for t in targets]
        rand = [d["top_n"] * d["per_target"][t]["gnn"]["n_actives"] / d["n_external_compounds"] for t in targets]
        ax2.barh(y - 0.19, full, 0.38, color="#1b6ca8", label="With soft edges")
        ax2.barh(y + 0.19, abl, 0.38, color="#9e9e9e", label="Without soft edges")
        ax2.vlines(rand, y - 0.42, y + 0.42, color="black", lw=1.4, label="Random")
        ax2.set_yticks(y, [d["driver_targets"][t] for t in targets])
        ax2.set_ylim(len(targets) - 0.5, -0.5)
        ax2.set_xlim(0, max(full + abl) + 0.5)
        ax2.xaxis.grid(True, color="#e6e6e6", lw=0.6)
        ax2.set_axisbelow(True)
        ax2.tick_params(axis="y", length=0)
        ax2.set_xlabel(f"Known actives in top {d['top_n']}")
        ax2.spines[["top", "right"]].set_visible(False)
        handles += ax2.get_legend_handles_labels()[0]

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False,
               handlelength=1.3, handleheight=0.9, columnspacing=1.2, fontsize=7.5)
    fig.canvas.draw()
    axes = [ax] + ([ax2] if ax2 is not None else [])
    top_y = max(axis.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y1
                for axis in axes)
    for axis, lab in zip(axes, "ab"):
        bb = axis.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.text(bb.x0, top_y + 0.01, lab, fontweight="bold", fontsize=11, va="bottom")

    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=300, bbox_inches="tight")
    print("saved", args.out, "picks:", [clean(names[c]["name"]) for c in picks])


if __name__ == "__main__":
    main()
