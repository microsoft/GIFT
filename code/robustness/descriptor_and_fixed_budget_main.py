"""Point 4 sensitivity: the same GIFT vs descriptor-only comparison for the main model checkpoints (the model version
behind the manuscript's lead-prioritization numbers, e.g. 47.9+/-4.2 driver top-10, 6.7+/-0.9 drivers at #1), read
from the saved results/gift/{run,descriptor_only_seed}_s/lead_prioritization.json (no new inference).
Writes results_main.json.
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import json
import os

import numpy as np

from descriptor_and_fixed_budget import DRIVERS, REPO, SC, SEEDS, msd, wil, lead_metrics

HERE = "results/robustness/descriptor_fixed_budget"
VARS = (("per_target", "gnn"), ("family", "gnn_family"), ("global", "gnn_shared"))


def from_json(path):
    L = json.load(open(path))["per_target"]
    out = {}
    for var, key in VARS:
        per = {g: {"roc_auc": L[t][key]["roc_auc"], "top10": L[t][key]["hits_top"],
                   "at1": L[t][key]["recovery_curve"][0]} for t, g in DRIVERS.items() if t in L}
        out[var] = {"roc_auc_mean": float(np.mean([v["roc_auc"] for v in per.values()])),
                    "top10_sum": int(sum(v["top10"] for v in per.values())),
                    "top10_mean": float(np.mean([v["top10"] for v in per.values()])),
                    "n_at1": int(sum(v["at1"] for v in per.values())), "n_drivers": len(per), "per_driver": per}
    return out


def main():
    res = {tag: {s: from_json(f"{REPO}/results/gift/{d}{s}/lead_prioritization.json") for s in SEEDS}
           for tag, d in (("gift", "run_"), ("descriptor_only", "descriptor_only_seed_"))}
    agg = {tag: {var: {k: msd([res[tag][s][var][k] for s in SEEDS])
                       for k in ("roc_auc_mean", "top10_sum", "top10_mean", "n_at1")} for var, _ in VARS}
           for tag in res}
    tests = {var: {k: wil([res["gift"][s][var][k] for s in SEEDS], [res["descriptor_only"][s][var][k] for s in SEEDS])
                   for k in ("roc_auc_mean", "top10_sum", "n_at1")} for var, _ in VARS}
    out = {}
    for var, key in VARS:
        per = {g: {"roc_auc": L[t][key]["roc_auc"], "top10": L[t][key]["hits_top"],
                   "at1": L[t][key]["recovery_curve"][0]} for t, g in DRIVERS.items() if t in L}
        out[var] = {"roc_auc_mean": float(np.mean([v["roc_auc"] for v in per.values()])),
                    "top10_sum": int(sum(v["top10"] for v in per.values())),
                    "top10_mean": float(np.mean([v["top10"] for v in per.values()])),
                    "n_at1": int(sum(v["at1"] for v in per.values())), "n_drivers": len(per), "per_driver": per}
    return out


def main():
    res = {tag: {s: from_json(f"{REPO}/results/gift/{d}{s}/lead_prioritization.json") for s in SEEDS}
           for tag, d in (("gift", "run_"), ("descriptor_only", "descriptor_only_seed_"))}
    agg = {tag: {var: {k: msd([res[tag][s][var][k] for s in SEEDS])
                       for k in ("roc_auc_mean", "top10_sum", "top10_mean", "n_at1")} for var, _ in VARS}
           for tag in res}
    tests = {var: {k: wil([res["gift"][s][var][k] for s in SEEDS], [res["descriptor_only"][s][var][k] for s in SEEDS])
                   for k in ("roc_auc_mean", "top10_sum", "n_at1")} for var, _ in VARS}
    out = {"source": "results/gift/*/lead_prioritization.json", "aggregate": agg,
           "wilcoxon_gift_vs_descriptor_only": tests, "per_seed": res}
    json.dump(out, open(f"{HERE}/results_main.json", "w"), indent=1)
    for var, _ in VARS:
        for k in ("roc_auc_mean", "top10_mean", "top10_sum", "n_at1"):
            g, d = agg["gift"][var][k], agg["descriptor_only"][var][k]
            print(f"{var:10s} {k:13s} GIFT {g['mean']:.3f}±{g['sd']:.3f}  DESC {d['mean']:.3f}±{d['sd']:.3f}  "
                  f"p={tests[var].get(k, float('nan')):.2g}")


if __name__ == "__main__":
    main()
