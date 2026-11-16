"""Point 3b: soft-edge recovery of held-out interactions, split into TEST-only and VALIDATION-only positives.

Read-only; CPU; no inference. Uses the saved Phase-2 soft-edge score matrices of the 30 main runs
(results/gift/run_{s}/soft_edge_proposals_scores.npy, copied from results/gift/baseline_models/drugban/run_{s} by run_main model_seed.sh and
produced by soft_edge_proposals.py from results/backbones/backbone_seed_{s}/best_model_phase2.pth) and reproduces the
statistic of soft_edge_proposals.py: among the k = 200,000 highest-scoring compound-protein pairs that are not
training interactions, count held-out interactions; enrichment = precision@k / base rate.
Variants
  pooled     : positives = val + test (as in the manuscript; must reproduce soft_edge_proposals.json)
  test_excl  : positives = test; validation positives removed from the candidate set   (primary test-only)
  test_incl  : positives = test; validation positives kept as candidates (counted as non-positives)
  val_excl   : positives = val;  test positives removed from the candidate set
Significance: one-sided hypergeometric P(X >= observed) (population = candidates, successes = positives,
draws = k); counted at p < 0.05 raw and after Bonferroni over 30 runs.
Usage: python code/robustness/softedge_recovery_test_only.py
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import json
import pickle

import numpy as np
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score

R = "results/gift"
K = 200000
OUT = "results/robustness/measured_activity_thresholds/pt3_softedge.json"


def stat(S, train, pos, drop):
    C, P = S.shape
    cand = np.ones(C * P, bool)
    for c, p in train | drop:
        cand[c * P + p] = False
    flat = np.where(cand)[0]
    order = flat[np.argsort(-S.ravel()[flat], kind="stable")]
    y = np.zeros(C * P, bool)
    for c, p in pos:
        y[c * P + p] = True
    top = order[:K]
    x, N, Kp = int(y[top].sum()), int(flat.size), int(y[flat].sum())
    exp = K * Kp / N
    return {"candidates": N, "positives": Kp, "in_top_k": x, "expected": exp, "enrichment": x / exp,
            "p_hyper": float(hypergeom.sf(x - 1, N, Kp, K)),
            "roc_auc": float(roc_auc_score(y[flat], S.ravel()[flat]))}


def main():
    rows = {}
    for s in range(42, 72):
        S = np.load(f"{R}/run_{s}/soft_edge_proposals_scores.npy")
        g = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
        so = g["split_override"]
        tr = {tuple(map(int, p)) for p in so["train"]}
        va = {tuple(map(int, p)) for p in so["val"]}
        te = {tuple(map(int, p)) for p in so["test"]}
        saved = json.load(open(f"{R}/run_{s}/soft_edge_proposals.json"))["global"]
        r = {"pooled": stat(S, tr, va | te, set()), "test_excl": stat(S, tr, te, va),
             "test_incl": stat(S, tr, te, set()), "val_excl": stat(S, tr, va, te)}
        r["pooled_matches_saved"] = (r["pooled"]["in_top_k"] == saved["held_out_in_top_k"]
                                     and abs(r["pooled"]["roc_auc"] - saved["roc_auc_all_candidates"]) < 1e-6)
        rows[s] = r
        print(s, {k: (v["in_top_k"], round(v["expected"], 1), round(v["enrichment"], 2)) for k, v in r.items()
                  if isinstance(v, dict)}, r["pooled_matches_saved"], flush=True)

    # "Interaction-like" runs as defined in the manuscript: the 10 runs with >= 2-fold pooled enrichment.
    il = [s for s in rows if rows[s]["pooled"]["enrichment"] >= 2]
    agg = {"interaction_like_runs_pooled_ge2x": il, "all_pooled_match_saved": all(r["pooled_matches_saved"] for r in rows.values())}
    for v in ("pooled", "test_excl", "test_incl", "val_excl"):
        e = np.array([rows[s][v]["enrichment"] for s in rows])
        x = np.array([rows[s][v]["in_top_k"] for s in rows])
        ex = np.array([rows[s][v]["expected"] for s in rows])
        p = np.array([rows[s][v]["p_hyper"] for s in rows])
        auc = np.array([rows[s][v]["roc_auc"] for s in rows])
        ei = np.array([rows[s][v]["enrichment"] for s in il])
        xi = np.array([rows[s][v]["in_top_k"] for s in il])
        auci = np.array([rows[s][v]["roc_auc"] for s in il])
        agg[v] = {"max_enrichment": float(e.max()), "n_runs_enrichment_ge2": int((e >= 2).sum()),
                  "n_runs_p_lt_0.05": int((p < 0.05).sum()), "n_runs_p_lt_0.05_bonf30": int((p < 0.05 / 30).sum()),
                  "expected_range": [float(ex.min()), float(ex.max())],
                  "il_runs_in_top_k_range": [int(xi.min()), int(xi.max())],
                  "il_runs_enrichment_range": [float(ei.min()), float(ei.max())],
                  "il_runs_roc_auc_range": [float(auci.min()), float(auci.max())],
                  "other_runs_in_top_k_range": [int(min(rows[s][v]["in_top_k"] for s in rows if s not in il)),
                                                int(max(rows[s][v]["in_top_k"] for s in rows if s not in il))],
                  "in_top_k_all_runs_range": [int(x.min()), int(x.max())],
                  "roc_auc_all_runs_mean_sd": [float(auc.mean()), float(auc.std(ddof=1))]}
    json.dump({"per_seed": rows, "agg": agg}, open(OUT, "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
