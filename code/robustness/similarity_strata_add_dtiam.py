"""Point 13: adds DTIAM to the finished per-seed similarity-strata outputs (pt13_out/seed_S.{json,npz}) without
re-scoring the other models. Uses the stored pairs, labels and strata of each seed and DTIAM's saved external scores
(results/robustness/dtiam/runs/run_S/scores.npz), asserting the pair order is identical. Same metrics as
pt13_similarity_strata.py. Usage: python pt13_add_dtiam.py
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

D = "results/robustness/similarity_strata"
OUT = D
DT = "results/robustness/dtiam/runs/run_{s}/scores.npz"

for s in range(42, 72):
    z = dict(np.load(f"{OUT}/seed_{s}.npz"))
    d = np.load(DT.format(s=s))
    assert np.array_equal(d["external|pairs"], z["pairs"]) and np.array_equal(d["external|y"], z["y"]), s
    sc, y, strat = d["external|score"].astype(float), z["y"], z["strat"]
    res = json.load(open(f"{OUT}/seed_{s}.json"))
    res["overall"]["DTIAM"] = {"roc_auc": float(roc_auc_score(y, sc)), "pr_auc": float(average_precision_score(y, sc))}
    for b, name in enumerate(("low", "mid", "high")):
        ix = strat == b
        res["strata"][name]["DTIAM"] = {"roc_auc": float(roc_auc_score(y[ix], sc[ix])),
                                        "pr_auc": float(average_precision_score(y[ix], sc[ix]))}
    json.dump(res, open(f"{OUT}/seed_{s}.json", "w"), indent=1)
    z["score_DTIAM"] = sc
    np.savez_compressed(f"{OUT}/seed_{s}.npz", **z)
print("added DTIAM to 30 seeds")
