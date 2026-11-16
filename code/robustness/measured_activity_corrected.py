"""Point 2: Table S15 recomputed under the definition stated in the text, plus a leakage audit of its negatives.

Definitions (per run, seed 42-71; positives as in eval_measured_activity.py: held-out pairs, median pChEMBL >= 5):
  stated_lt5     : recorded pairs WITH a pChEMBL value, median < 5, not training, in the evaluation reserve
                   (the definition the text states; asserted identical to the [3,5) 'moderate' set)
  stated_3_5     : as above with 3 <= median < 5 (= Table S15 'moderate')
  measured_lt5_all_reserve_free : all 564 with-pChEMBL pairs of median < 5, not training, ignoring the reserve
                   (sensitivity only: 459 lie outside the reserve and may have been sampled as GIFT/RF training
                   negatives at their base rate)
  original_all   : the published Table S15 'all measured' set (2,257 = 105 with pChEMBL + 2,152 recorded pairs
                   WITHOUT pChEMBL whose median defaults to 0.0 in all_known_pchembl)
Leakage audit: every negative set is checked against (i) training positives, (ii) val/test positives, (iii) the
training-negative pool (non-reserve pairs: training_exclusion blocks the reserve), (iv) the GIFT validation
negatives used for checkpoint selection (sampled with evaluation_exclusion(all_known_pairs, reserve), which blocks
every recorded pair) and (v) graph compound-protein edges.
Inference only (frozen results/gift/run_{s}/e2e GIFT checkpoints; RF refit exactly as eval_measured_activity.py).
Usage: python code/robustness/measured_activity_corrected.py [--seeds ...] [--device cuda]
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.getcwd())
import pipeline_utils as vs  # noqa: E402
from graph_cleaning import clean_graph  # noqa: E402
from eval_measured_activity import per_target_roc, pooled_metrics  # noqa: E402
from evaluate_chembl35 import load_gift_model  # noqa: E402
from negative_pool import evaluation_exclusion, reserved_eval_mask, sample_negatives, training_exclusion  # noqa: E402

D = "results/robustness/measured_activity_thresholds"
CACHE = f"{D}/cache"
OUT = f"{D}/pt2_s15_corrected.json"
KEYS = ["per_target_roc_auc", "mcc", "pr_auc", "roc_auc"]


def one_seed(s, dev):
    v5 = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
    med = {tuple(map(int, k)): float(v) for k, v in dict(v5["all_known_pchembl"]).items()}
    has_pc = {tuple(map(int, k)) for k in v5["pchembl_n_records"]}
    so = v5["split_override"]
    tr = {tuple(map(int, p)) for p in so["train"]}
    va = {tuple(map(int, p)) for p in so["val"]}
    te = {tuple(map(int, p)) for p in so["test"]}
    nc, npr = int(v5["num_compounds"]), int(v5["num_proteins"])
    R = reserved_eval_mask(nc, npr)
    pos = {"pooled": sorted(p for p in va | te if med.get(p, 0.0) >= 5.0),
           "test": sorted(p for p in te if med.get(p, 0.0) >= 5.0),
           "val": sorted(p for p in va if med.get(p, 0.0) >= 5.0)}
    negs = {
        "stated_lt5": sorted(p for p in has_pc if med[p] < 5.0 and p not in tr and R[p]),
        "stated_3_5": sorted(p for p in has_pc if 3.0 <= med[p] < 5.0 and p not in tr and R[p]),
        "measured_lt5_all_reserve_free": sorted(p for p in has_pc if med[p] < 5.0 and p not in tr),
        "original_all": sorted(p for p in med if med[p] < 5.0 and p not in tr and R[p]),
    }
    moderate_orig = sorted(p for p in med if 3.0 <= med[p] < 5.0 and p not in tr and R[p])
    assert negs["stated_3_5"] == moderate_orig

    # Leakage audit.
    cp = np.asarray(v5["cp_edge_index"])
    cp_pairs = set()
    if cp.size:
        a, b = cp[0].astype(int), cp[1].astype(int)
        for u, w in zip(a, b):
            if u < nc and nc <= w < nc + npr:
                cp_pairs.add((int(u), int(w - nc)))
            elif w < nc and nc <= u < nc + npr:
                cp_pairs.add((int(w), int(u - nc)))
    known = {tuple(map(int, p)) for p in v5["all_known_pairs"]}
    val_pairs = sorted(va)
    rng = np.random.RandomState(s)
    val_negs = set(sample_negatives(rng, np.asarray(sorted({c for c, _ in val_pairs})), npr, 5 * len(val_pairs),
                                    evaluation_exclusion(known | tr | va | te, R)))
    audit = {}
    for nn, ng in negs.items():
        sg = set(ng)
        audit[nn] = {"n": len(ng), "in_train_pos": len(sg & tr), "in_val_test_pos": len(sg & (va | te)),
                     "outside_reserve_training_negative_pool": int(sum(not R[p] for p in ng)),
                     "in_gift_val_negatives": len(sg & val_negs), "in_graph_cp_edges": len(sg & cp_pairs),
                     "with_pchembl": int(sum(p in has_pc for p in ng)),
                     "n_lt3": int(sum(med[p] < 3 for p in ng if p in has_pc))}

    model, _ = load_gift_model(v5, dev, f"results/gift/run_{s}/e2e/trained_model.pth")
    gg = clean_graph(v5)[0]
    train_pairs = sorted(tr)
    rf_negs = sample_negatives(np.random.RandomState(s), np.arange(nc), npr, len(train_pairs),
                               training_exclusion(train_pairs, R))
    feat = vs.build_pair_feature_fn(v5)
    X = np.stack([feat(c, p) for c, p in train_pairs + rf_negs])
    yv = np.array([1] * len(train_pairs) + [0] * len(rf_negs))
    rf = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=8).fit(X, yv)
    allp = sorted(set(pos["pooled"]).union(*[set(v) for v in negs.values()]))
    sg = dict(zip(allp, vs.score_gnn_pairs(model, gg, allp, dev)))
    sr = dict(zip(allp, rf.predict_proba(np.stack([feat(c, p) for c, p in allp]))[:, 1]))
    res = {}
    for nn, neg in negs.items():
        for pn, ps in pos.items():
            pairs = ps + neg
            y = np.array([1] * len(ps) + [0] * len(neg))
            rec = {}
            for m, sc in (("gift", sg), ("rf", sr)):
                sv = np.array([sc[p] for p in pairs])
                pt, nt = per_target_roc(pairs, y, sv)
                rec[m] = {**pooled_metrics(y, sv), "per_target_roc_auc": pt, "n_targets_both": nt}
            res[f"{nn}|{pn}"] = rec
    return {"metrics": res, "audit": audit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 72)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--aggregate_only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev = torch.device(a.device)
    per = {}
    for s in a.seeds:
        f = f"{CACHE}/s15_corrected_seed_{s}.json"
        if not os.path.exists(f):
            if a.aggregate_only:
                continue
            json.dump(one_seed(s, dev), open(f, "w"))
        per[s] = json.load(open(f))
        print("seed", s, "done", flush=True)
    seeds = sorted(per)
    if len(seeds) < 2:
        return
    # Reproduction: original_all|pooled and stated_3_5|pooled must equal the published per-seed values.
    ref = json.load(open("results/gift/measured_activity.json"))["per_seed"]
    repro = max(abs(per[s]["metrics"][f"{v}|pooled"][m][k] - ref[str(s)][rv][rm][k])
                for s in seeds for v, rv in (("original_all", "all"), ("stated_3_5", "moderate"))
                for m, rm in (("gift", "gnn"), ("rf", "rf")) for k in KEYS)
    agg = {"seeds": seeds, "max_abs_diff_vs_published": repro, "audit": {}}
    for nn in per[seeds[0]]["audit"]:
        agg["audit"][nn] = {k: sorted({per[s]["audit"][nn][k] for s in seeds}) for k in per[seeds[0]]["audit"][nn]}
    for key in per[seeds[0]]["metrics"]:
        agg[key] = {}
        for m in ("gift", "rf"):
            agg[key][m] = {k: [float(np.mean([per[s]["metrics"][key][m][k] for s in seeds])),
                               float(np.std([per[s]["metrics"][key][m][k] for s in seeds], ddof=1))] for k in KEYS}
            for k in ("n_pos", "n_neg", "n_targets_both"):
                v = [per[s]["metrics"][key][m][k] for s in seeds]
                agg[key][m][k] = [int(min(v)), int(max(v))]
        agg[key]["wilcoxon_gift_vs_rf"] = {k: float(wilcoxon([per[s]["metrics"][key]["gift"][k] for s in seeds],
                                                             [per[s]["metrics"][key]["rf"][k] for s in seeds]).pvalue)
                                           for k in KEYS}
        agg[key]["gift_gt_rf_runs"] = {k: int(sum(per[s]["metrics"][key]["gift"][k] > per[s]["metrics"][key]["rf"][k]
                                                  for s in seeds)) for k in KEYS}
    json.dump({"agg": agg, "per_seed": per}, open(OUT, "w"), indent=1)
    print(json.dumps({k: agg[k] for k in ("max_abs_diff_vs_published", "audit")}, indent=1))


if __name__ == "__main__":
    main()
