"""Main-table rows and paired comparisons for the manuscript (tab:main, tab:dta, tab:confusion, results text).

Proposed model from --prop (e.g. results/gift); baselines are independent of the proposed run: RF from the
proposed run's eval files, GCN/SAGE/GAT from results/gift/baseline_models/gnn/gnn_baselines_std_seed_s.json, DeepDTA/GraphDTA from
results/gift/dta_leadprio/dta_metrics_seed_s.json, DrugBAN from results/gift/baseline_models/drugban/run_s/dta_baselines_drugban.json.
Counts (TP/FP/FN/TN) of the proposed model and the RF are derived from recall/precision and the benchmark sizes.
Usage: python paper_numbers.py --prop results/gift [--json out.json]
"""
import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

SEEDS = range(42, 72)
M = "gift"
BENCH = ["same_version", "same_version_matched", "external"]
METHODS = ["proposed", "rf", "gcn", "sage", "gat", "deepdta", "graphdta", "drugban"]
ROWS = [("PR-AUC", "auprc"), ("MCC", "mcc"), ("Precision", "precision"), ("F1", "f1"), ("ROC-AUC", "roc_auc"),
        ("Per-target ROC-AUC", "per_target_roc_auc"), ("Recall", "recall"), ("FP", "fp"), ("Total errors", "errors")]
LOWER = {"fp", "fn", "errors"}
COUNTS = {"tp", "fp", "fn", "tn", "errors"}


def counts(npos, nneg, m):
    tp = round(m["recall"] * npos)
    fp = round(tp / m["precision"] - tp) if tp else 0
    return {"tp": tp, "fn": npos - tp, "fp": fp, "tn": nneg - fp}


def load(prop, s, b):
    ban = json.load(open(f"results/gift/baseline_models/drugban/run_{s}/dta_baselines_drugban.json"))
    npos, nneg = ban["bench_sizes"][b]
    ev = json.load(open(f"{prop}/run_{s}/eval_final.json"))["results_by_method"]
    ex = json.load(open(f"{prop}/run_{s}/eval_external_matched.json"))["results"]
    gnn = json.load(open(f"results/gift/baseline_models/gnn/gnn_baselines_std_seed_{s}.json"))["results"]
    dta = json.load(open(f"results/gift/dta_leadprio/dta_metrics_seed_{s}.json"))["results"]
    gb = "cross_version" if b == "external" else b
    out = {
        "proposed": ex[M] if b == "external" else ev[M][b],
        "rf": ex["random_forest"] if b == "external" else ev["random_forest"][b],
        "gcn": gnn["gcn"][gb], "sage": gnn["sage"][gb], "gat": gnn["gat"][gb],
        "deepdta": dta["deepdta"][b], "graphdta": dta["graphdta"][b], "drugban": ban["results"]["drugban"][b],
    }
    for k, m in out.items():
        m = dict(m)
        if "fp" not in m:
            m.update(counts(npos, nneg, m))
        m["errors"] = m["fp"] + m["fn"]
        out[k] = m
    return out


def cell(v, k):
    m, s = np.mean(v), np.std(v, ddof=1)
    return f"{m:,.1f}$\\pm${s:,.1f}" if k in COUNTS else f"{m:.3f}$\\pm${s:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {}
    for b in BENCH:
        data = [load(a.prop, s, b) for s in SEEDS]
        res[b] = {}
        print(f"\n%% {b}")
        for lab, k in ROWS + [("FN", "fn"), ("TP", "tp"), ("TN", "tn")]:
            v = {n: np.array([d[n][k] for d in data], float) for n in METHODS}
            means = {n: float(np.mean(x)) for n, x in v.items()}
            best = (min if k in LOWER else max)(METHODS, key=lambda n: round(means[n], 1 if k in COUNTS else 3))
            cells = [cell(v[n], k) for n in METHODS]
            cells = [f"\\textbf{{{c}}}" if n == best else c for n, c in zip(METHODS, cells)]
            if (lab, k) in ROWS:
                arrow = "$\\downarrow$" if k in LOWER else "$\\uparrow$"
                print(f"{lab} {arrow} & " + " & ".join(cells) + " \\\\")
            cmp = {}
            for n in METHODS[1:]:
                better = (v["proposed"] < v[n]) if k in LOWER else (v["proposed"] > v[n])
                p = float(wilcoxon(v["proposed"], v[n]).pvalue) if np.any(v["proposed"] != v[n]) else 1.0
                cmp[n] = {"better": int(better.sum()), "p": p}
            others = np.stack([v[n] for n in METHODS[1:]])
            beat_all = ((v["proposed"] < others.min(0)) if k in LOWER else (v["proposed"] > others.max(0))).sum()
            res[b][k] = {"means": means, "sd": {n: float(np.std(x, ddof=1)) for n, x in v.items()},
                         "best": best, "vs": cmp, "beats_all_runs": int(beat_all)}
        for k in ["roc_auc", "mcc", "auprc", "precision", "errors", "recall", "per_target_roc_auc"]:
            r = res[b][k]
            print(f"   {k:20s} " + "  ".join(f"{n}:{r['vs'][n]['better']}/30 p={r['vs'][n]['p']:.1g}"
                                             for n in METHODS[1:]) + f"  beats-all {r['beats_all_runs']}/30")
        e = res[b]["errors"]["means"]
        nb = min((n for n in METHODS[1:]), key=e.get)
        print(f"   errors: proposed {e['proposed']:.1f}, next best {nb} {e[nb]:.1f}, "
              f"reduction {100 * (1 - e['proposed'] / e[nb]):.0f}%")
    iso = [json.load(open(f"{a.prop}/run_{s}/eval_final.json"))["results_by_method"][M]["same_version_isolated"]
           for s in SEEDS]
    print("\nisolated: " + ", ".join(f"{k} {np.mean([r[k] for r in iso]):.3f}$\\pm${np.std([r[k] for r in iso], ddof=1):.3f}"
                                     for k in ["roc_auc", "auprc", "precision", "recall", "f1", "mcc"]))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
