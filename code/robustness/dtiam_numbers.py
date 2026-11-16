"""DTIAM numbers for the manuscript tables, paired by seed with the main model (main model).

DTIAM (frozen BerMol + ESM-2 representations with its official AutoGluon DTI head) was trained in each of the 30 runs
under the shared DTI-baseline protocol (results/robustness/dtiam): the same exported training/validation pairs,
negative-sampling exclusion and seed as DeepDTA/GraphDTA/DrugBAN, scored by the same evaluator (--import_scores).

Prints, per benchmark, the LaTeX cells (mean$\\pm$sd) of DTIAM, the runs in which GIFT is better, paired Wilcoxon
p-values, the confusion-matrix row, and the driver-kinase lead-prioritization numbers. The JSON output has the same
layout as drugban_manuscript_numbers.py so that the table patcher can consume it.
Usage: python dtiam_manuscript_numbers.py [--json out.json]
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

SEEDS = range(42, 72)
PROP = "results/gift"
DT = "results/robustness/dtiam/runs"
BENCH = ["same_version", "same_version_matched", "external"]
METRICS = ["auprc", "mcc", "precision", "f1", "roc_auc", "per_target_roc_auc", "recall"]
COUNTS = ["fp", "fn", "errors", "tp", "tn"]
LOWER = {"fp", "fn", "errors"}
DRIVERS = ("EGFR", "ALK", "ROS1", "RET", "MET", "ERBB2", "BRAF", "NTRK1", "NTRK2", "NTRK3")


def counts(npos, nneg, m):
    tp = round(m["recall"] * npos)
    fp = round(tp / m["precision"] - tp) if tp else 0
    return {"tp": tp, "fn": npos - tp, "fp": fp, "tn": nneg - fp}


def cell(v, k):
    m, s = np.mean(v), np.std(v, ddof=1)
    if k in COUNTS:
        return f"{m:,.1f}$\\pm${s:,.1f}"
    return f"{m:.3f}$\\pm${s:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    seeds = [s for s in SEEDS if os.path.exists(f"{DT}/run_{s}/dta_baselines_dtiam.json")]
    print(f"DTIAM runs available: {len(seeds)}/30")
    out = {"n": len(seeds), "bench": {}, "lead": {}}
    for b in BENCH:
        rows = {"proposed": [], "dtiam": []}
        for s in seeds:
            dt = json.load(open(f"{DT}/run_{s}/dta_baselines_dtiam.json"))
            npos, nneg = dt["bench_sizes"][b]
            if b == "external":
                p = json.load(open(f"{PROP}/run_{s}/eval_external_matched.json"))["results"]["gift"]
            else:
                p = json.load(open(f"{PROP}/run_{s}/eval_final.json"))["results_by_method"]["gift"][b]
            p = dict(p, **counts(npos, nneg, p))
            for name, m in (("proposed", p), ("dtiam", dt["results"]["dtiam"][b])):
                rows[name].append(dict(m, errors=m["fp"] + m["fn"]))
        res = {}
        print(f"\n## {b}")
        for k in METRICS + COUNTS:
            v = {n: np.array([r[k] for r in rows[n]], dtype=float) for n in rows}
            better = (v["proposed"] < v["dtiam"]) if k in LOWER else (v["proposed"] > v["dtiam"])
            p = float(wilcoxon(v["proposed"], v["dtiam"]).pvalue) if np.any(v["proposed"] != v["dtiam"]) else 1.0
            res[k] = {"cell": cell(v["dtiam"], k), "proposed_better": int(better.sum()), "wilcoxon_p": p,
                      "means": {n: float(np.mean(x)) for n, x in v.items()}}
            print(f"{k:20s} DTIAM {cell(v['dtiam'], k):28s} GIFT {cell(v['proposed'], k):28s} "
                  f"GIFT better {better.sum():2d}/{len(seeds)} p={p:.1e}")
        out["bench"][b] = res

    at1, top10, per_driver = [], [], {d: [] for d in DRIVERS}
    for s in seeds:
        pd = json.load(open(f"{DT}/run_{s}/lead_dtiam.json"))["lead"]["dtiam"]["per_driver"]
        at1.append(sum(pd[d]["active_at_1"] for d in DRIVERS if d in pd))
        top10.append(sum(pd[d]["hits_top10"] for d in DRIVERS if d in pd))
        for d in DRIVERS:
            per_driver[d].append(pd[d]["hits_top10"] if d in pd else np.nan)
    out["lead"] = {"at1": [float(np.mean(at1)), float(np.std(at1, ddof=1))],
                   "top10": [float(np.mean(top10)), float(np.std(top10, ddof=1))],
                   "per_driver_top10": {d: [float(np.nanmean(v)), float(np.nanstd(v, ddof=1))]
                                        for d, v in per_driver.items()},
                   "at1_per_seed": at1, "top10_per_seed": top10}
    print(f"\nlead: #1 {np.mean(at1):.2f}+-{np.std(at1, ddof=1):.2f}  top10 {np.mean(top10):.1f}+-{np.std(top10, ddof=1):.1f}")
    for d in DRIVERS:
        print(f"  {d:6s} top10 {out['lead']['per_driver_top10'][d][0]:.2f}")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
