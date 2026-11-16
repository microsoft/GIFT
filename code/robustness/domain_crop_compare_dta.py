"""Point 6: effect of domain-preserving sequence crops on the DeepDTA/GraphDTA baselines.

Compares, per model, the domain-crop retrain (dta_crop) against the N-terminal-prefix retrain (dta_nocrop, the
reproducibility control that matches the published truncation) on the external benchmark (ROC-AUC, PR-AUC, per-target
ROC-AUC, MCC) and the driver-kinase shortlist (actives in the top 10 and at rank 1, summed over the ten drivers),
over the seeds available in both. Paired Wilcoxon over the shared seeds. Also checks that dta_nocrop reproduces the
published dta_leadprio metrics. Read-only; writes compare_crop_nocrop.json.
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

D = "results/robustness/domain_crop"
MODELS = ("deepdta", "graphdta")
DRIVERS = ("EGFR", "ALK", "ROS1", "RET", "MET", "ERBB2", "BRAF", "NTRK1", "NTRK2", "NTRK3")
EXT_KEYS = ("roc_auc", "auprc", "per_target_roc_auc", "mcc")


def seeds_in(d):
    return sorted(int(f.split("_seed_")[1].split(".")[0]) for f in glob.glob(f"{d}/dta_metrics_seed_*.json"))


def ext(d, s, m):
    r = json.load(open(f"{d}/dta_metrics_seed_{s}.json"))["results"][m]["external"]
    return {k: r[k] for k in EXT_KEYS}


def lead(d, s, m):
    pd = json.load(open(f"{d}/dta_lead_seed_{s}.json"))["lead"][m]["per_driver"]
    return {"top10": sum(pd[k]["hits_top10"] for k in DRIVERS if k in pd),
            "at1": sum(pd[k]["active_at_1"] for k in DRIVERS if k in pd)}


def main():
    crop_s = set(seeds_in(f"{D}/dta_crop"))
    nocrop_s = set(seeds_in(f"{D}/dta_nocrop"))
    shared = sorted(crop_s & nocrop_s)
    out = {"seeds_crop": sorted(crop_s), "seeds_nocrop": sorted(nocrop_s), "shared_seeds": shared, "models": {}}
    for m in MODELS:
        rec = {"external": {}, "driver": {}}
        for tag, metric in (("roc_auc", lambda d, s: ext(d, s, m)["roc_auc"]),
                            ("auprc", lambda d, s: ext(d, s, m)["auprc"]),
                            ("per_target_roc_auc", lambda d, s: ext(d, s, m)["per_target_roc_auc"]),
                            ("mcc", lambda d, s: ext(d, s, m)["mcc"])):
            c = [metric(f"{D}/dta_crop", s) for s in shared]
            n = [metric(f"{D}/dta_nocrop", s) for s in shared]
            p = wilcoxon(c, n).pvalue if len(shared) >= 2 and any(a != b for a, b in zip(c, n)) else None
            rec["external"][tag] = {"crop": [float(np.mean(c)), float(np.std(c, ddof=1))],
                                    "nocrop": [float(np.mean(n)), float(np.std(n, ddof=1))], "wilcoxon_p": p}
        for tag in ("top10", "at1"):
            c = [lead(f"{D}/dta_crop", s, m)[tag] for s in shared]
            n = [lead(f"{D}/dta_nocrop", s, m)[tag] for s in shared]
            p = wilcoxon(c, n).pvalue if len(shared) >= 2 and any(a != b for a, b in zip(c, n)) else None
            rec["driver"][tag] = {"crop": [float(np.mean(c)), float(np.std(c, ddof=1))],
                                  "nocrop": [float(np.mean(n)), float(np.std(n, ddof=1))], "wilcoxon_p": p}
        out["models"][m] = rec
    json.dump(out, open(f"{D}/compare_crop_nocrop.json", "w"), indent=1)
    print(f"shared seeds ({len(shared)}): {shared}")
    for m in MODELS:
        print(f"\n== {m} ==")
        for k, v in out["models"][m]["external"].items():
            print(f"  ext {k:<20} crop {v['crop'][0]:.3f}±{v['crop'][1]:.3f}  nocrop {v['nocrop'][0]:.3f}±{v['nocrop'][1]:.3f}  p={v['wilcoxon_p']}")
        for k, v in out["models"][m]["driver"].items():
            print(f"  driver {k:<17} crop {v['crop'][0]:.2f}±{v['crop'][1]:.2f}  nocrop {v['nocrop'][0]:.2f}±{v['nocrop'][1]:.2f}  p={v['wilcoxon_p']}")


if __name__ == "__main__":
    main()
