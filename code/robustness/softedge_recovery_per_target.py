"""Point 3b (addendum): within-target soft-edge ROC-AUC (supplement: 'within a target, mean ROC-AUC 0.88 +- 0.08'
over the 10 interaction-like runs) with pooled, TEST-only and VALIDATION-only held-out positives.

Read-only; uses saved results/gift/run_{s}/soft_edge_proposals_scores.npy. Statistic copied from
soft_edge_proposals.py: per protein, rank graph compounds without a training interaction at it; targets with
>= 3 held-out positives. test_excl / val_excl drop the other held-out split from the candidates.
Usage: python code/robustness/softedge_recovery_per_target.py
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import json
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

R = "results/gift"
OUT = "results/robustness/measured_activity_thresholds/pt3_softedge_pertarget.json"
IL = [47, 48, 50, 51, 52, 54, 56, 60, 61, 65]


def per_target(S, train, pos, drop):
    C, P = S.shape
    T = np.zeros((C, P), bool)
    for c, p in train | drop:
        T[c, p] = True
    Y = np.zeros((C, P), bool)
    for c, p in pos:
        Y[c, p] = True
    aucs = []
    for p in range(P):
        m = ~T[:, p]
        y = Y[m, p]
        if y.sum() >= 3:
            aucs.append(roc_auc_score(y, S[m, p]))
    return {"n_targets": len(aucs), "mean_roc_auc": float(np.mean(aucs)) if aucs else float("nan")}


def main():
    rows = {}
    for s in range(42, 72):
        S = np.load(f"{R}/run_{s}/soft_edge_proposals_scores.npy")
        so = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))["split_override"]
        tr, va, te = ({tuple(map(int, p)) for p in so[k]} for k in ("train", "val", "test"))
        saved = json.load(open(f"{R}/run_{s}/soft_edge_proposals.json"))["per_target_summary"]
        r = {"pooled": per_target(S, tr, va | te, set()), "test_excl": per_target(S, tr, te, va),
             "val_excl": per_target(S, tr, va, te)}
        r["pooled_matches_saved"] = abs(r["pooled"]["mean_roc_auc"] - saved["mean_roc_auc"]) < 1e-5  # regenerated scores differ by float noise
        rows[s] = r
        print(s, r, flush=True)
    agg = {"all_pooled_match_saved": all(r["pooled_matches_saved"] for r in rows.values())}
    for v in ("pooled", "test_excl", "val_excl"):
        a = np.array([rows[s][v]["mean_roc_auc"] for s in IL])
        n = [rows[s][v]["n_targets"] for s in IL]
        agg[v] = {"il_mean_sd": [float(a.mean()), float(a.std(ddof=1))], "il_n_targets_range": [min(n), max(n)]}
    json.dump({"per_seed": rows, "agg": agg}, open(OUT, "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
