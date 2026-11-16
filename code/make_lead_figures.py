"""Lead-prioritization figures of the paper (GIFT vs DeepDTA, GraphDTA, DrugBAN, DTIAM; 30 runs).

Figure 3 (lead_prioritization_gift): (a) per-target ROC-AUC of GIFT over the 252 targets (mean of 30 runs); (b) known
actives in the top 10 per driver kinase, GIFT vs baselines and random (bars: mean over 30 runs; error bars: s.d.).
Figure 2 (lead_top_picks_gift): stacked panels with aligned driver-kinase columns. (a) GIFT drug rankings (seed 42) with
activity evidence; (b) known actives in the top ten per method and kinase (mean over 30 runs).
Needs the Liberation Sans font (apt install fonts-liberation) to reproduce the layout exactly.
Usage: python code/make_lead_figures.py [--prop results/gift] [--outdir figures]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from make_top_picks_figure import DRIVER_ORDER, EVIDENCE, clean

SEEDS = range(42, 72)
EVIDENCE_B = {k: (lab, "#C8C8C8" if k == "untested" else col) for k, (lab, col) in EVIDENCE.items()}
EVIDENCE_A = {k: (lab, EVIDENCE[k][1]) for k, lab in (("active", "Known active compounds in external benchmark"),
              ("active_parent", "Additional parent-compound activity"),
              ("tested", "Recorded activity, threshold not met"), ("untested", "No activity record"))}
KINASE_PATH = "data/protein_kinase_annotation.json"
# Okabe-Ito colorblind-safe palette; blue is reserved for GIFT.
METHODS = [("gift", "GIFT", "#0072B2"), ("deepdta", "DeepDTA", "#E69F00"),
           ("graphdta", "GraphDTA", "#009E73"), ("drugban", "DrugBAN", "#CC79A7"), ("dtiam", "DTIAM", "#D55E00")]
DTIAM = "results/robustness/dtiam/runs/run_{s}/lead_dtiam.json"
GIFT_BLUE = METHODS[0][2]
WIDTH = 6.5  # inches; included at \textwidth so fonts print at their nominal size (7 pt).
LEGEND_GAP = 0.27  # vertical gap (axes fraction) between the x-axis label and the key below it.
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "mathtext.default": "regular", "pdf.fonttype": 42, "savefig.dpi": 600})


def load_hits(prop):
    """hits[method][gene] -> array over seeds of known actives among the 10 top-ranked drugs."""
    hits = {m: {g: [] for g in DRIVER_ORDER} for m, _, _ in METHODS}
    for s in SEEDS:
        d = json.load(open(f"{prop}/run_{s}/lead_prioritization.json"))
        for t, g in d["driver_targets"].items():
            hits["gift"][g].append(d["per_target"][t]["gnn"]["hits_top"])
        dta = json.load(open(f"{prop}/dta_leadprio/dta_lead_seed_{s}.json"))["lead"]
        ban = json.load(open(f"results/gift/baseline_models/drugban/run_{s}/drugban_dgl/lead_drugban.json"))["lead"]["drugban"]
        dti = json.load(open(DTIAM.format(s=s)))["lead"]["dtiam"]
        for g in DRIVER_ORDER:
            hits["deepdta"][g].append(dta["deepdta"]["per_driver"][g]["hits_top10"])
            hits["graphdta"][g].append(dta["graphdta"]["per_driver"][g]["hits_top10"])
            hits["drugban"][g].append(ban["per_driver"][g]["hits_top10"])
            hits["dtiam"][g].append(dti["per_driver"][g]["hits_top10"])
    return {m: {g: np.array(v, float) for g, v in per.items()} for m, per in hits.items()}


def p_label(p):
    if p < 1e-3:
        return f"$p<10^{{{int(np.ceil(np.log10(p)))}}}$"
    return f"$p={p:.3f}$"


def panel_labels(fig, axes, letters="abc"):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(r).transformed(fig.transFigure.inverted()) for ax in axes]
    top = max(b.y1 for b in boxes) + 0.01
    for b, lab in zip(boxes, letters):
        fig.text(b.x0, top, lab, fontweight="bold", fontsize=9, va="bottom")


def load_top1(prop):
    """Number of driver kinases (of 10) with a known active ranked first, per run and method."""
    top1 = {m: [] for m, _, _ in METHODS}
    for s in SEEDS:
        d = json.load(open(f"{prop}/run_{s}/lead_prioritization.json"))
        sc, lab = d["driver_scores"], d["driver_labels"]
        top1["gift"].append(sum(max(sc[t], key=sc[t].get) in lab[t] for t in d["driver_targets"]))
        dta = json.load(open(f"{prop}/dta_leadprio/dta_lead_seed_{s}.json"))["lead"]
        top1["deepdta"].append(dta["deepdta"]["drivers_active_at_1"])
        top1["graphdta"].append(dta["graphdta"]["drivers_active_at_1"])
        top1["drugban"].append(json.load(open(f"results/gift/baseline_models/drugban/run_{s}/drugban_dgl/lead_drugban.json"))
                               ["lead"]["drugban"]["drivers_active_at_1"])
        top1["dtiam"].append(sum(v["active_at_1"] for v in
                                 json.load(open(DTIAM.format(s=s)))["lead"]["dtiam"]["per_driver"].values()))
    return {m: np.array(v, float) for m, v in top1.items()}


def bar_panel(ax, values, ylabel, rand, step):
    """Bars = mean over runs, points = individual runs, brackets = paired Wilcoxon GIFT vs each baseline."""
    rng = np.random.default_rng(0)
    for k, (m, lab, c) in enumerate(METHODS):
        v = values[m]
        ax.bar(k, v.mean(), 0.66, color=c, edgecolor="none", zorder=2)
        ax.scatter(k + rng.uniform(-0.2, 0.2, len(v)), v, s=3, color="black", alpha=0.35, lw=0, zorder=3)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar(k, v.mean(), yerr=ci, color="black", lw=0.8, capsize=2, capthick=0.8, zorder=4)
    ax.axhline(rand, color="black", ls="--", lw=0.6, zorder=4)
    ax.text(len(METHODS) - 0.4, rand, f"Random\n{rand:.1f}", ha="left", va="center", fontsize=6, clip_on=False)
    top = max(v.max() for v in values.values())
    for k, (m, lab, _) in enumerate(METHODS[1:], start=1):
        p = wilcoxon(values["gift"], values[m]).pvalue
        win, loss = int((values["gift"] > values[m]).sum()), int((values["gift"] < values[m]).sum())
        print(f"  GIFT vs {lab}: p={p:.2e}, higher in {win}, lower in {loss} of 30 runs")
        yb = top + step * (0.6 + (k - 1))
        ax.plot([0, 0, k, k], [yb - 0.22 * step, yb, yb, yb - 0.22 * step], color="black", lw=0.5)
        ax.text(k / 2, yb + 0.05 * step, f"{p_label(p)}  ({win}\u2013{loss})", ha="center", va="bottom",
                fontsize=5.8)
    ax.set_xticks(range(len(METHODS)), [f"{lab}\n{values[m].mean():.1f}" for m, lab, _ in METHODS])
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.set_ylim(0, top + step * len(METHODS))
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color="#e3e3e3", lw=0.4)
    ax.set_axisbelow(True)


def figure_leadprio(prop, hits, out):
    """Figure 3: (a) per-target ROC-AUC of GIFT over the 252 targets (mean of 30 runs); (b) known actives in the top 10
    per driver kinase, GIFT vs DeepDTA, GraphDTA, DrugBAN (mean +- s.d. over 30 runs)."""
    d = json.load(open(f"{prop}/run_42/lead_prioritization.json"))
    pt = d["per_target"]
    n_ext, top_n = d["n_external_compounds"], d["top_n"]
    gene_to_t = {g: t for t, g in d["driver_targets"].items()}
    n_act = {g: pt[gene_to_t[g]]["gnn"]["n_actives"] for g in DRIVER_ORDER}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 2.6), gridspec_kw={"width_ratios": [1, 1.55],
                                                                           "wspace": 0.42})
    # (a) histogram of the per-target ROC-AUC averaged over the 30 runs, kinase vs other targets (stacked).
    kin = json.load(open(KINASE_PATH))
    targets = list(pt)
    runs = []
    for s in SEEDS:
        pts = json.load(open(f"{prop}/run_{s}/lead_prioritization.json"))["per_target"]
        assert set(pts) == set(targets), f"seed {s}: target set differs from seed 42"
        runs.append([pts[t]["gnn"]["roc_auc"] for t in targets])
    runs = np.array(runs)
    auc = runs.mean(0)
    n_above = (runs > 0.5).sum(1)
    is_kin = np.array([kin.get(t, {}).get("kinase", False) for t in targets])
    ax1.hist([auc[is_kin], auc[~is_kin]], bins=np.linspace(0, 1, 21), stacked=True,
             color=[GIFT_BLUE, "#A6CEE3"], edgecolor="white", lw=0.8, label=["Kinase targets", "Other targets"])
    ax1.axvline(0.5, color="#595959", lw=0.8, ls="--")
    ax1.axvline(auc.mean(), color="black", lw=1.4)
    ymax = ax1.get_ylim()[1] * 1.12
    ax1.set_ylim(0, ymax)
    ax1.text(0.49, ymax * 0.5, "Random", ha="right", va="center", color="#595959", fontsize=6.5)
    ax1.text(auc.mean() + 0.015, ymax * 0.92, f"Mean {auc.mean():.3f}", ha="left", va="center", fontsize=6.5)
    ax1.set_xlim(0, 1)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax1.set_xlabel("Per-target ROC-AUC, GIFT (mean of 30 runs)")
    ax1.set_ylabel("Number of targets")
    ax1.set_title(f"Above random for {n_above.mean():.0f}$\\pm${n_above.std(ddof=1):.0f} of {len(targets)} targets",
                  pad=3)
    ax1.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.2)
    print(f"(a) 30 runs: mean per-target ROC-AUC {runs.mean(1).mean():.3f}+-{runs.mean(1).std(ddof=1):.3f}, "
          f"above 0.5 for {n_above.mean():.1f}+-{n_above.std(ddof=1):.1f}/{len(targets)}; "
          f"targets with run-mean ROC-AUC > 0.5: {(auc > 0.5).sum()}")

    # (b) per-kinase grouped bars (mean over 30 runs, error bars = s.d.), sorted by the GIFT mean.
    genes = sorted(DRIVER_ORDER, key=lambda g: -hits["gift"][g].mean())
    h = 0.8 / len(METHODS)
    for k, (m, lab, c) in enumerate(METHODS):
        mu = np.array([hits[m][g].mean() for g in genes])
        sd = np.array([hits[m][g].std(ddof=1) for g in genes])
        y = np.arange(len(genes)) + (k - (len(METHODS) - 1) / 2) * h
        ax2.barh(y, mu, h, xerr=sd, color=c, edgecolor="white", lw=0.3, label=lab, zorder=3,
                 error_kw={"lw": 0.5, "capsize": 0, "ecolor": "#333333"})
    rand = [top_n * n_act[g] / n_ext for g in genes]
    ax2.vlines(rand, np.arange(len(genes)) - 0.45, np.arange(len(genes)) + 0.45, color="black", lw=1.2,
               label="Random", zorder=5)
    ax2.set_yticks(range(len(genes)), [f"{g} ({n_act[g]})" for g in genes])
    ax2.set_ylim(len(genes) - 0.5, -0.5)
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlim(0, 9)
    ax2.set_xticks(range(10))
    ax2.xaxis.grid(True, color="#e3e3e3", lw=0.5)
    ax2.set_axisbelow(True)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ax2.set_xlabel(f"Known actives in top {top_n} (mean \u00b1 s.d., 30 runs)")
    hs, ls = ax2.get_legend_handles_labels()
    order = [ls.index(lab) for _, lab, _ in METHODS] + [ls.index("Random")]
    ax2.legend([hs[k] for k in order], [ls[k] for k in order], frameon=False, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, -LEGEND_GAP), handlelength=1.6, columnspacing=1.6, borderaxespad=0)
    for g in genes:
        print(f"  {g:6s} " + "  ".join(f"{lab} {hits[m][g].mean():.1f}+-{hits[m][g].std(ddof=1):.1f}"
                                       for m, lab, _ in METHODS))
    panel_labels(fig, (ax1, ax2))
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    print("saved", out)


_MEASURE = {}


def text_width(text, size, weight="normal", style="normal"):
    """Width in inches of a single-line text at the given font size."""
    if "fig" not in _MEASURE:
        _MEASURE["fig"] = plt.figure(dpi=100)
    fig = _MEASURE["fig"]
    t = fig.text(0, 0, text, fontsize=size, fontweight=weight, fontstyle=style)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def wrap(text, width, size, weight="normal"):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if cur and text_width(trial, size, weight) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur]


def figure_toppicks(prop, hits, out, names_path, web_path, top=5, fs=7.0, rh=0.11):
    """Figure 2, two panels side by side sharing the driver-kinase rows (one set of kinase names, on the left).
    (a) GIFT drug rankings (run with seed 42): columns are the drugs ranked among the five highest for at least one
        kinase; numbers give the rank among the 95 drugs, outlines mark ranks 1-5, colors give the activity evidence;
        a count column gives the known active drugs out of 95 for each kinase.
    (b) Percentage of the ten highest-ranked drugs with known activity (mean over 30 runs) per method, a Random column
        (expected for a random ranking), and a Mean row over the kinases; shading = known actives in the top ten (0-10).
        Bold italic = highest mean among the five methods in each row (unrounded, ties marked); * = GIFT above every baseline
        (two-sided paired Wilcoxon, Holm-corrected over the 40 kinase-baseline comparisons, p < 0.05).
    The layout is set in inches on a single axes so that rows, columns, and labels align exactly."""
    d = json.load(open(f"{prop}/run_42/lead_prioritization.json"))
    names = json.load(open(names_path))["compounds"]
    web = json.load(open(web_path))
    gene_to_t = {g: t for t, g in d["driver_targets"].items()}
    genes = [g for g in DRIVER_ORDER if g in gene_to_t]
    targets = [gene_to_t[g] for g in genes]
    n_act = {g: d["per_target"][gene_to_t[g]]["gnn"]["n_actives"] for g in genes}
    n_ext, top_n = d["n_external_compounds"], d["top_n"]
    scores = d["driver_scores"]
    ranks = {t: {c: r + 1 for r, c in enumerate(sorted(scores[t], key=lambda c: -scores[t][c]))} for t in targets}

    def evidence(c, t):
        if c in d["driver_labels"][t]:
            return "active"
        rec = web.get(c, {}).get("targets", {}).get(t)
        if rec and (rec["max_pchembl"] or 0) >= 5:
            return "active_parent"
        return "tested" if rec else "untested"

    drugs = {c for t in targets for c, r in ranks[t].items() if r <= top}
    drugs = sorted(drugs, key=lambda c: (np.mean([ranks[t][c] for t in targets]), clean(names[c]["name"])))
    drug_names = [clean(names[c]["name"]) for c in drugs]

    tests = [(g, m, 1.0 if np.all(hits["gift"][g] == hits[m][g]) else wilcoxon(hits["gift"][g], hits[m][g]).pvalue)
             for g in genes for m, _, _ in METHODS[1:]]
    holm, running = {}, 0.0
    for i, k in enumerate(np.argsort([t[2] for t in tests])):
        running = max(running, (len(tests) - i) * tests[k][2])
        holm[(tests[k][0], tests[k][1])] = min(1.0, running)
    beats_all = {g: all(holm[(g, m)] < 0.05 and hits["gift"][g].mean() > hits[m][g].mean() for m, _, _ in METHODS[1:])
                 for g in genes}
    print("(b) GIFT > all baselines (Holm p<0.05):", [g for g in genes if beats_all[g]])

    # (b) values: mean known actives in the top ten (unrounded), per kinase row and the Mean row.
    cols_b = [(m, lab) for m, lab, _ in METHODS] + [("random", "Random")]
    rows_b = genes + ["Mean"]
    val = {}
    for g in genes:
        for m, _ in cols_b:
            val[(g, m)] = top_n * n_act[g] / n_ext if m == "random" else float(hits[m][g].mean())
    for m, _ in cols_b:
        val[("Mean", m)] = float(np.mean([val[(g, m)] for g in genes]))
    best = {r: max(val[(r, m)] for m, _, _ in METHODS) for r in rows_b}
    is_best = {(r, m): m != "random" and abs(val[(r, m)] - best[r]) < 1e-9 for r in rows_b for m, _ in cols_b}
    print("(b) bold (highest mean):", {r: [m for m, _ in cols_b if is_best[(r, m)]] for r in rows_b})

    # ---- geometry (inches) ----
    W = WIDTH
    fs_small, fs_title, fs_letter = fs - 0.6, fs + 1.0, fs + 3.0
    lab_w = max(text_width(g, fs, "bold") for g in genes) + 0.08          # kinase-name column
    cnt_w = max(text_width("out of 95", fs_small), text_width("26/95", fs)) + 0.10
    cw_a = 0.235
    gap = max(text_width(g, fs, "bold") for g in genes + ["Mean"]) + 0.16   # kinase names repeated for (b)
    x_lab = lab_w
    x_cnt = lab_w + 0.02
    x_a = x_cnt + cnt_w + 0.04
    x_b = x_a + len(drugs) * cw_a + gap
    c45 = 0.7071
    cw_b = (W - 0.03 - text_width(cols_b[-1][1], fs) * c45 - x_b) / (len(cols_b) - 0.5)
    head_a = (max(text_width(n, fs) for n in drug_names) + fs / 72) * c45 + 0.06       # labels at 45 degrees
    head_b = (max(text_width(lab, fs) for _, lab in cols_b) + fs / 72) * c45 + 0.06
    lt, ls = fs_title / 72 * 1.25, fs_small / 72 * 1.3
    title_a = "GIFT drug rankings for each target"
    sub_a = wrap("One run, seed 42; cell numbers are ranks among 95 drugs.", len(drugs) * cw_a + cnt_w + lab_w - 0.2,
                 fs_small)
    tb_x = x_b - gap + 0.02
    title_b = wrap("Percentage of top-10 drugs with known activity at each target", W - tb_x - 0.2, fs_title, "bold")
    sub_b = wrap("Mean over 30 runs; random expectation shown for comparison.", W - tb_x - 0.2, fs_small)
    block_a = lt + len(sub_a) * ls + 0.05 + head_a
    block_b = len(title_b) * lt + len(sub_b) * ls + 0.05 + head_b
    y_grid = max(block_a, block_b) + 0.02
    y_mean = y_grid + len(genes) * rh + 0.04
    y_end_a = y_grid + len(genes) * rh
    y_end_b = y_mean + rh
    leg_row = 0.13
    y_leg = y_end_a + 0.10
    y_cb = y_end_b + 0.10
    H = max(y_leg + 3 * leg_row, y_cb + 0.07 + 0.30) + 0.02

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    txt = dict(fontsize=fs, ha="center", va="center")

    # Titles and subtitles, placed so each panel's column headers end just above the shared grid.
    y = y_grid - 0.02 - block_a
    ax.text(0, y, "a", fontsize=fs_letter, fontweight="bold", va="top", ha="left")
    ax.text(0.17, y + 0.015, title_a, fontsize=fs_title, fontweight="bold", va="top", ha="left")
    for k, line in enumerate(sub_a):
        ax.text(0.17, y + lt + k * ls + 0.01, line, fontsize=fs_small, color="#444444", va="top", ha="left")
    y = y_grid - 0.02 - block_b
    ax.text(tb_x, y, "b", fontsize=fs_letter, fontweight="bold", va="top", ha="left")
    for k, line in enumerate(title_b):
        ax.text(tb_x + 0.17, y + 0.015 + k * lt, line, fontsize=fs_title, fontweight="bold", va="top", ha="left")
    for k, line in enumerate(sub_b):
        ax.text(tb_x + 0.17, y + len(title_b) * lt + k * ls + 0.01, line, fontsize=fs_small, color="#444444",
                va="top", ha="left")

    # Shared kinase names (one set, on the left) and the count column.
    for i, g in enumerate(genes):
        yc = y_grid + (i + 0.5) * rh
        ax.text(x_lab, yc, g, fontsize=fs, fontweight="bold", ha="right", va="center")
        ax.text(x_cnt + cnt_w / 2, yc, f"{n_act[g]}/{n_ext}", color="#333333", **txt)
    for k, line in enumerate(reversed(["Known", "active drugs", "out of 95"])):
        ax.text(x_cnt + cnt_w / 2, y_grid - 0.04 - k * fs_small / 72 * 1.2, line, fontsize=fs_small, color="#333333",
                ha="center", va="bottom")
    for i, g in enumerate(genes):
        ax.text(x_b - 0.05, y_grid + (i + 0.5) * rh, g, fontsize=fs, fontweight="bold", ha="right", va="center")
    ax.text(x_b - 0.05, y_mean + rh / 2, "Mean", fontsize=fs, fontweight="bold", ha="right", va="center")

    # (a) rank grid.
    for j, name in enumerate(drug_names):
        ax.text(x_a + (j + 0.5) * cw_a - 0.02, y_grid - 0.03, name, fontsize=fs, rotation=45, rotation_mode="anchor",
                ha="left", va="bottom")
    for i, t in enumerate(targets):
        for j, c in enumerate(drugs):
            x, y = x_a + j * cw_a, y_grid + i * rh
            r = ranks[t][c]
            ax.add_patch(plt.Rectangle((x, y), cw_a, rh, facecolor=EVIDENCE_A[evidence(c, t)][1], edgecolor="white",
                                       lw=0.8))
            if r <= top:
                ax.add_patch(plt.Rectangle((x + 0.012, y + 0.012), cw_a - 0.024, rh - 0.024, fill=False,
                                           edgecolor="black", lw=0.9, zorder=4))
            ax.text(x + cw_a / 2, y + rh / 2, str(r), zorder=5, fontweight="bold" if r <= top else "normal",
                    color="black" if r <= top else "#444444", **txt)

    # (b) percentage grid; shading by the number of known actives (0-10).
    cmap, norm = plt.get_cmap("Blues"), matplotlib.colors.Normalize(0, 10)
    for j, (m, lab) in enumerate(cols_b):
        ax.text(x_b + (j + 0.5) * cw_b - 0.02, y_grid - 0.03, lab, fontsize=fs, rotation=45, rotation_mode="anchor",
                ha="left", va="bottom")
    for i, r in enumerate(rows_b):
        y = y_mean if r == "Mean" else y_grid + i * rh
        for j, (m, _) in enumerate(cols_b):
            v = val[(r, m)]
            x = x_b + j * cw_b
            ax.add_patch(plt.Rectangle((x, y), cw_b, rh, facecolor=cmap(norm(v)), edgecolor="white", lw=0.8))
            star = "*" if (m == "gift" and r != "Mean" and beats_all[r]) else ""
            ax.text(x + cw_b / 2, y + rh / 2, f"{100 * v / top_n:.0f}%{star}",
                    fontweight="bold" if is_best[(r, m)] else "normal", fontstyle="italic" if is_best[(r, m)] else "normal",
                    color="white" if v > 5.5 else "black", **txt)

    # Legend of (a), under panel a.
    handles = [(EVIDENCE_A[k][1], None, EVIDENCE_A[k][0]) for k in ("active", "active_parent", "tested", "untested")]
    handles.append(("white", "black", "Top 1\u20135"))
    col_x = [0.0, 0.0]
    col_x[1] = 0.17 + max(text_width(lab, fs_small) for _, _, lab in handles[:3]) + 0.18
    for k, (fc, ec, lab) in enumerate(handles):
        cx, cy = col_x[k // 3], y_leg + (k % 3) * leg_row
        ax.add_patch(plt.Rectangle((cx, cy), 0.13, 0.085, facecolor=fc, edgecolor=ec or "none", lw=0.9))
        ax.text(cx + 0.17, cy + 0.0425, lab, fontsize=fs_small, ha="left", va="center")

    # Color scale of (b), under panel b, with the 0-10 scale of known actives.
    cb_x0, cb_x1 = x_b, x_b + len(cols_b) * cw_b
    n = 200
    for k in range(n):
        ax.add_patch(plt.Rectangle((cb_x0 + k * (cb_x1 - cb_x0) / n, y_cb), (cb_x1 - cb_x0) / n * 1.02, 0.07,
                                   facecolor=cmap(norm(10 * (k + 0.5) / n)), edgecolor="none"))
    for v in range(0, 11, 2):
        x = cb_x0 + v / 10 * (cb_x1 - cb_x0)
        ax.plot([x, x], [y_cb + 0.07, y_cb + 0.095], color="black", lw=0.6)
        ax.text(x, y_cb + 0.10, str(v), fontsize=fs_small, ha="center", va="top")
    ax.text((cb_x0 + cb_x1) / 2, y_cb + 0.10 + fs_small / 72 * 1.25, "Known actives in top ten", fontsize=fs_small,
            ha="center", va="top")

    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=600)
    print("saved", out, f"({W:.2f} x {H:.2f} in)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", default="results/gift")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--names", default="data/chembl_names.json")
    ap.add_argument("--web", default="data/chembl37_driver_activities.json")
    a = ap.parse_args()
    hits = load_hits(a.prop)
    for m, lab, _ in METHODS:
        print(f"{lab:9s} mean known actives in top 10 over drivers: "
              f"{np.mean([hits[m][g].mean() for g in DRIVER_ORDER]):.2f}")
    figure_leadprio(a.prop, hits, os.path.join(a.outdir, "lead_prioritization_gift"))
    figure_toppicks(a.prop, hits, os.path.join(a.outdir, "lead_top_picks_gift"), a.names, a.web)


if __name__ == "__main__":
    main()
