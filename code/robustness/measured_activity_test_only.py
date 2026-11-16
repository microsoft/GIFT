"""Point 3a: Table S15 (measured active vs measured below-threshold) with TEST-only and VALIDATION-only positives.

Inference only (frozen GIFT e2e checkpoints of results/gift/run_{s}, CPU) plus the same RF refit as
eval_measured_activity.py (100 trees, random_state=seed, 1:1 training negatives). Pair selection is copied
from eval_measured_activity.py; positives are split by split_override['val'] / ['test'].
The pooled variant must reproduce results/gift/measured_activity.json (checked and reported).
Usage: python code/robustness/measured_activity_test_only.py [--seeds ...]
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
from negative_pool import reserved_eval_mask, sample_negatives, training_exclusion  # noqa: E402

OUT = "results/robustness/measured_activity_thresholds/pt3_measured_activity.json"
CACHE = "results/robustness/measured_activity_thresholds/cache"
KEYS = ["per_target_roc_auc", "mcc", "pr_auc", "roc_auc", "precision", "recall"]


def one_seed(s, dev):
    v5 = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
    model, _ = load_gift_model(v5, dev, f"results/gift/run_{s}/e2e/trained_model.pth")
    gg = clean_graph(v5)[0]
    med = {tuple(map(int, k)): float(v) for k, v in dict(v5["all_known_pchembl"]).items()}
    so = v5["split_override"]
    tr = {tuple(map(int, p)) for p in so["train"]}
    va = {tuple(map(int, p)) for p in so["val"]}
    te = {tuple(map(int, p)) for p in so["test"]}
    nc, npr = int(v5["num_compounds"]), int(v5["num_proteins"])
    R = reserved_eval_mask(nc, npr)
    pos = {"pooled": sorted(p for p in va | te if med.get(p, 0.0) >= 5.0),
           "test": sorted(p for p in te if med.get(p, 0.0) >= 5.0),
           "val": sorted(p for p in va if med.get(p, 0.0) >= 5.0)}
    negs = {"moderate": sorted(p for p in med if 3.0 <= med[p] < 5.0 and p not in tr and R[p[0], p[1]]),
            "all": sorted(p for p in med if med[p] < 5.0 and p not in tr and R[p[0], p[1]])}
    negs["no_pchembl_only"] = [p for p in negs["all"] if med[p] == 0.0]

    train_pairs = sorted(tr)
    rf_negs = sample_negatives(np.random.RandomState(s), np.arange(nc), npr, len(train_pairs),
                               training_exclusion(train_pairs, R))
    feat = vs.build_pair_feature_fn(v5)
    X = np.stack([feat(c, p) for c, p in train_pairs + rf_negs])
    yv = np.array([1] * len(train_pairs) + [0] * len(rf_negs))
    rf = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=16).fit(X, yv)

    allp = sorted(set(pos["pooled"]) | set(negs["all"]))
    sg = dict(zip(allp, vs.score_gnn_pairs(model, gg, allp, dev)))
    sr = dict(zip(allp, rf.predict_proba(np.stack([feat(c, p) for c, p in allp]))[:, 1]))

    out = {}
    for nn, neg in negs.items():
        for pn, ps in pos.items():
            pairs = ps + neg
            y = np.array([1] * len(ps) + [0] * len(neg))
            rec = {}
            for m, sc in (("gnn", sg), ("rf", sr)):
                sv = np.array([sc[p] for p in pairs])
                pt, nt = per_target_roc(pairs, y, sv)
                rec[m] = {**pooled_metrics(y, sv), "per_target_roc_auc": pt, "n_targets_both": nt}
            out[f"{nn}|{pn}"] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 72)))
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    per = {}
    os.makedirs(CACHE, exist_ok=True)
    for s in a.seeds:
        f = f"{CACHE}/measured_activity_seed_{s}.json"
        if os.path.exists(f):
            per[s] = json.load(open(f))
        else:
            per[s] = one_seed(s, dev)
            json.dump(per[s], open(f, "w"))
        g = per[s]["all|test"]["gnn"]
        print(f"seed {s}: all|test gnn ptROC={g['per_target_roc_auc']:.3f} MCC={g['mcc']:.3f} "
              f"pos={g['n_pos']} neg={g['n_neg']}", flush=True)

    ref = json.load(open("results/gift/measured_activity.json"))["per_seed"]
    repro = {}
    for s in a.seeds:
        for v in ("moderate", "all"):
            for m in ("gnn", "rf"):
                d = max(abs(per[s][f"{v}|pooled"][m][k] - ref[str(s)][v][m][k]) for k in KEYS)
                repro[f"{s}|{v}|{m}"] = d
    agg = {"max_abs_diff_pooled_vs_saved": {"gnn": max(v for k, v in repro.items() if k.endswith("gnn")),
                                            "rf": max(v for k, v in repro.items() if k.endswith("rf"))}}
    for key in per[a.seeds[0]]:
        agg[key] = {}
        for m in ("gnn", "rf"):
            agg[key][m] = {}
            for k in KEYS:
                v = np.array([per[s][key][m][k] for s in a.seeds])
                agg[key][m][k] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]
            for k in ("n_pos", "n_neg", "n_targets_both"):
                v = [per[s][key][m][k] for s in a.seeds]
                agg[key][m][k] = [int(min(v)), int(max(v))]
        if len(a.seeds) >= 5:
            agg[key]["wilcoxon_gnn_vs_rf"] = {k: float(wilcoxon([per[s][key]["gnn"][k] for s in a.seeds],
                                                                [per[s][key]["rf"][k] for s in a.seeds]).pvalue)
                                              for k in ("per_target_roc_auc", "mcc")}
            agg[key]["gnn_gt_rf_runs"] = {k: int(sum(per[s][key]["gnn"][k] > per[s][key]["rf"][k] for s in a.seeds))
                                          for k in ("per_target_roc_auc", "mcc")}
    json.dump({"per_seed": per, "agg": agg}, open(OUT, "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
