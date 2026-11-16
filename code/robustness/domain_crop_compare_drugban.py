"""Point 6: effect of the domain-preserving sequence crop on DrugBAN (official DGL implementation).

Compares the domain-crop retrain (drugban_crop) against the N-terminal-prefix retrain (drugban_nocrop, the control that
matches the published truncation) on the external benchmark (ROC-AUC, PR-AUC, per-target ROC-AUC, MCC) and the
driver-kinase shortlist (actives in the top 10 and at rank 1, summed over the ten drivers), over the seeds finished in
both. Paired Wilcoxon over the shared seeds. Also reports the published DrugBAN runs (results/gift/baseline_models/drugban) on the same seeds as
a reproducibility check of drugban_nocrop. Read-only; writes compare_drugban_crop_nocrop.json.
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import json
import os

import numpy as np
from scipy.stats import wilcoxon

D = "results/robustness/domain_crop"
REPO = _REPO
DRIVERS = ("EGFR", "ALK", "ROS1", "RET", "MET", "ERBB2", "BRAF", "NTRK1", "NTRK2", "NTRK3")
EXT_KEYS = ("roc_auc", "auprc", "per_target_roc_auc", "mcc")
SRC = {"crop": f"{D}/drugban_crop/run_{{s}}", "nocrop": f"{D}/drugban_nocrop/run_{{s}}",
       "published": f"{REPO}/results/gift/baseline_models/drugban/run_{{s}}"}


def done(tmpl, s):
    d = tmpl.format(s=s)
    return os.path.isfile(f"{d}/dta_baselines_drugban.json") and os.path.isfile(f"{d}/drugban_dgl/lead_drugban.json")


def metrics(tmpl, s):
    d = tmpl.format(s=s)
    r = json.load(open(f"{d}/dta_baselines_drugban.json"))["results"]["drugban"]
    pd = json.load(open(f"{d}/drugban_dgl/lead_drugban.json"))["lead"]["drugban"]["per_driver"]
    out = {k: r["external"][k] for k in EXT_KEYS}
    out["top10"] = sum(pd[k]["hits_top10"] for k in DRIVERS if k in pd)
    out["at1"] = sum(pd[k]["active_at_1"] for k in DRIVERS if k in pd)
    out["best_epoch"] = r.get("best_epoch")
    return out


def summary(a, b):
    p = wilcoxon(a, b).pvalue if len(a) >= 2 and any(x != y for x, y in zip(a, b)) else None
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return {"a": [float(np.mean(a)), sd(a)], "b": [float(np.mean(b)), sd(b)], "wilcoxon_p": p}


def main():
    seeds = range(42, 72)
    shared = [s for s in seeds if done(SRC["crop"], s) and done(SRC["nocrop"], s)]
    pub = [s for s in shared if done(SRC["published"], s)]
    m = {k: {s: metrics(SRC[k], s) for s in (pub if k == "published" else shared)} for k in SRC}
    out = {"shared_seeds": shared, "crop_vs_nocrop": {}, "nocrop_vs_published": {"seeds": pub}}
    for key in EXT_KEYS + ("top10", "at1"):
        out["crop_vs_nocrop"][key] = summary([m["crop"][s][key] for s in shared], [m["nocrop"][s][key] for s in shared])
        if pub:
            out["nocrop_vs_published"][key] = summary([m["nocrop"][s][key] for s in pub],
                                                      [m["published"][s][key] for s in pub])
    json.dump(out, open(f"{D}/compare_drugban_crop_nocrop.json", "w"), indent=1)
    print(f"shared seeds ({len(shared)}): {shared}")
    print("== DrugBAN crop (a) vs nocrop (b) ==")
    for k, v in out["crop_vs_nocrop"].items():
        print(f"  {k:<20} crop {v['a'][0]:.3f}±{v['a'][1]:.3f}  nocrop {v['b'][0]:.3f}±{v['b'][1]:.3f}  p={v['wilcoxon_p']}")
    if pub:
        print(f"== reproducibility: nocrop (a) vs published shared baseline protocol (b), {len(pub)} seeds ==")
        for k, v in out["nocrop_vs_published"].items():
            if k != "seeds":
                print(f"  {k:<20} nocrop {v['a'][0]:.3f}  published {v['b'][0]:.3f}  p={v['wilcoxon_p']}")


if __name__ == "__main__":
    main()
