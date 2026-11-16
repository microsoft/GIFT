"""Supplementary figures showing the value of GIFT over every baseline (30 runs each).

lead_value_gift (Figure S5): driver-kinase lead prioritization of the 95 external drugs.
  (a) Known actives among the 10 top-ranked drugs, summed over the ten driver kinases, per run and method;
  (b) driver kinases (of 10) with a known active ranked first, per run and method.
  Each baseline is annotated with the number of runs in which GIFT is higher / lower.
dti_value_gift (Figure S2): DTI prediction on the three benchmarks of increasing difficulty.
  (a) MCC and (b) PR-AUC of all nine methods (mean +- s.d.), GIFT highlighted;
  (c) runs (of 30) in which GIFT exceeds each baseline in PR-AUC, precision and MCC on each benchmark.
Usage (repo root): python code/make_value_figures.py [--outdir figures]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import make_lead_figures as base
from make_top_picks_figure import DRIVER_ORDER

SEEDS = base.SEEDS
DTI = base.METHODS  # GIFT, DeepDTA, GraphDTA, DrugBAN, DTIAM (Okabe-Ito colors, GIFT blue)
ALL = [("gift", "GIFT", "#0072B2"), ("rf", "RF", "#999999"), ("gcn", "GCN", "#B3B3B3"), ("sage", "SAGE", "#8C8C8C"),
       ("gat", "GAT", "#666666"), ("deepdta", "DeepDTA", "#E69F00"), ("graphdta", "GraphDTA", "#009E73"),
       ("drugban", "DrugBAN", "#CC79A7"), ("dtiam", "DTIAM", "#D55E00")]
BENCH = [("same_version", "Random\nnegatives"), ("same_version_matched", "Target\nidentification"),
         ("external", "Drugs outside\nthe graph")]
SHORT = ["Random neg.", "Target ID", "Unseen drugs"]
KEYS = [("auprc", "PR-AUC"), ("precision", "Precision"), ("mcc", "MCC")]


def strip_panel(ax, vals, ylabel, rand=None):
    rng = np.random.default_rng(0)
    for k, (m, lab, c) in enumerate(DTI):
        v = vals[m]
        ax.boxplot(v, positions=[k], widths=0.55, showfliers=False, patch_artist=True, zorder=2,
                   boxprops={"facecolor": c, "alpha": 0.25, "edgecolor": c, "lw": 0.8},
                   medianprops={"color": c, "lw": 1.6}, whiskerprops={"color": c, "lw": 0.8},
                   capprops={"color": c, "lw": 0.8})
        ax.scatter(k + rng.uniform(-0.17, 0.17, len(v)), v, s=6, color=c, lw=0, alpha=0.9, zorder=3)
    top = max(v.max() for v in vals.values())
    for k, (m, lab, _) in enumerate(DTI[1:], start=1):
        win, loss = int((vals["gift"] > vals[m]).sum()), int((vals["gift"] < vals[m]).sum())
        ax.text(k, top * 1.05, f"{win}/30", ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=DTI[0][2])
    ax.text(0, top * 1.05, "GIFT higher\nin runs:", ha="center", va="bottom", fontsize=5.6)
    if rand is not None:
        ax.axhline(rand, color="black", ls="--", lw=0.7, zorder=1)
        ax.text(-0.55, rand + top * 0.015, f"Random ({rand:.1f})", ha="left", va="bottom", fontsize=5.8)
    ax.set_xticks(range(len(DTI)), [f"{lab}\n{vals[m].mean():.1f}" for m, lab, _ in DTI])
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_xlim(-0.6, len(DTI) - 0.4)
    ax.set_ylim(0, top * 1.22)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color="#e3e3e3", lw=0.4)
    ax.set_axisbelow(True)


def lead_value(outdir):
    hits = base.load_hits("results/gift")
    top10 = {m: np.sum([hits[m][g] for g in DRIVER_ORDER], axis=0) for m, _, _ in DTI}
    top1 = base.load_top1("results/gift")
    d = json.load(open("results/gift/run_42/lead_prioritization.json"))
    n_ext, top_n = d["n_external_compounds"], d["top_n"]
    rand10 = sum(top_n * d["per_target"][t]["gnn"]["n_actives"] / n_ext for t in d["driver_targets"])
    rand1 = sum(d["per_target"][t]["gnn"]["n_actives"] / n_ext for t in d["driver_targets"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(base.WIDTH, 2.6), gridspec_kw={"wspace": 0.3})
    strip_panel(ax1, top10, "Known actives in top 10\n(summed over 10 driver kinases)", rand10)
    strip_panel(ax2, top1, "Driver kinases with a known\nactive ranked first (of 10)", rand1)
    base.panel_labels(fig, (ax1, ax2))
    out = os.path.join(outdir, "lead_value_gift")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    for name, v in (("top10", top10), ("top1", top1)):
        print(name, {lab: f"{v[m].mean():.2f} (GIFT higher {int((v['gift'] > v[m]).sum())}/30)" for m, lab, _ in DTI},
              f"random {rand10 if name == 'top10' else rand1:.2f}")
    print("saved", out)


def per_run_metrics():
    R = {m: {b: {k: [] for k, _ in KEYS} for b, _ in BENCH} for m, _, _ in ALL}
    for s in SEEDS:
        ev = json.load(open(f"results/gift/run_{s}/eval_final.json"))["results_by_method"]["gift"]
        ex = json.load(open(f"results/gift/run_{s}/eval_external_matched.json"))["results"]["gift"]
        rf = json.load(open(f"results/gift/baseline_models/rf/rf_seed_{s}.json"))["benchmarks"]
        gnn = json.load(open(f"results/gift/baseline_models/gnn/gnn_baselines_std_seed_{s}.json"))["results"]
        dta = json.load(open(f"results/gift/dta_leadprio/dta_metrics_seed_{s}.json"))["results"]
        ban = json.load(open(f"results/gift/baseline_models/drugban/run_{s}/dta_baselines_drugban.json"))["results"]["drugban"]
        dti = json.load(open(f"results/robustness/dtiam/runs/run_{s}/dta_baselines_dtiam.json"))["results"]["dtiam"]
        for b, _ in BENCH:
            gk = "cross_version" if b == "external" else b
            row = {"gift": ex if b == "external" else ev[b], "rf": rf[b], "gcn": gnn["gcn"][gk], "sage": gnn["sage"][gk],
                   "gat": gnn["gat"][gk], "deepdta": dta["deepdta"][b], "graphdta": dta["graphdta"][b],
                   "drugban": ban[b], "dtiam": dti[b]}
            for m, r in row.items():
                for k, _ in KEYS:
                    R[m][b][k].append(r[k])
    return {m: {b: {k: np.array(v) for k, v in d.items()} for b, d in bd.items()} for m, bd in R.items()}


def dti_value(outdir):
    R = per_run_metrics()
    fig = plt.figure(figsize=(base.WIDTH, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.48)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    x = np.arange(len(BENCH))
    for ax, (k, lab) in zip(axes[:2], (("mcc", "MCC"), ("auprc", "PR-AUC"))):
        for m, name, c in ALL[::-1]:
            mu = np.array([R[m][b][k].mean() for b, _ in BENCH])
            sd = np.array([R[m][b][k].std(ddof=1) for b, _ in BENCH])
            gift = m == "gift"
            ax.errorbar(x, mu, yerr=sd, color=c, lw=2.2 if gift else 0.9, ms=4.5 if gift else 3, marker="o",
                        capsize=0, elinewidth=0.6, zorder=6 if gift else 3, label=name)
        ax.set_xticks(x, [t for _, t in BENCH], fontsize=6.2)
        ax.set_xlim(-0.3, len(BENCH) - 0.7)
        ax.set_ylim(0, 1)
        ax.set_ylabel(lab)
        ax.yaxis.grid(True, color="#e3e3e3", lw=0.4)
        ax.set_axisbelow(True)
    best = {b: max((R[m][b]["mcc"].mean(), m) for m, _, _ in ALL[1:])[0] for b, _ in BENCH}
    for i, (b, _) in enumerate(BENCH):
        g = R["gift"][b]["mcc"].mean()
        axes[0].text(i, g + 0.05, f"+{100 * (g / best[b] - 1):.0f}%", ha="center", va="bottom", fontsize=6,
                     color=ALL[0][2], fontweight="bold")
    # (c) runs in which GIFT beats each baseline.
    rows = [(m, name) for m, name, _ in ALL[1:]]
    cols = [(b, k) for b, _ in BENCH for k, _ in KEYS]
    M = np.array([[int((R["gift"][b][k] > R[m][b][k]).sum()) for b, k in cols] for m, _ in rows])
    ax = axes[2]
    ax.imshow(M, cmap="Blues", vmin=0, vmax=30, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=5.6,
                    color="white" if M[i, j] >= 20 else "black")
    ax.set_yticks(range(len(rows)), [n for _, n in rows], fontsize=6.2)
    ax.set_xticks(range(len(cols)), [l for _ in BENCH for _, l in KEYS], rotation=90, fontsize=5.8)
    for j in (3, 6):
        ax.axvline(j - 0.5, color="white", lw=2)
    for j, t in enumerate(SHORT):
        ax.text(3 * j + 1, -0.75, t, ha="center", va="bottom", fontsize=5.8)
    ax.set_title("Runs (of 30) in which GIFT is higher", pad=16, fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs[::-1], ls[::-1], frameon=False, ncol=9, loc="upper center", bbox_to_anchor=(0.36, -0.02),
               handlelength=1.4, columnspacing=1.0, fontsize=6)
    base.panel_labels(fig, axes, letters="abc")
    out = os.path.join(outdir, "dti_value_gift")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    print("MCC margin over best baseline:", {b: f"{100 * (R['gift'][b]['mcc'].mean() / best[b] - 1):.0f}%" for b, _ in BENCH})
    print("min runs GIFT higher (PR-AUC/precision/MCC, all baselines, all benchmarks):", int(M.min()))
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    lead_value(a.outdir)
    dti_value(a.outdir)


if __name__ == "__main__":
    main()
