"""Save the random-forest baseline of one run (seed) and check it reproduces the reported RF results.

The RF baseline (scikit-learn defaults: 100 trees, random_state=seed) is trained on the same descriptors, training
interactions, and training negatives in evaluate_chembl35.py (ChEMBL 35 benchmarks) and
evaluate_external.py (external benchmark); both build the identical model, so one file per seed covers
the three benchmarks. The model is refitted here, saved with joblib, and its metrics are compared with the saved ones.
Usage: python train_rf_baseline.py --seed 42 [--prop results/gift] [--out results/gift/baseline_models/rf]
"""

import argparse
import json
import os
import pickle

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import pipeline_utils as vs
from evaluate_chembl35 import RF_NEG_RATIO
from evaluate_external import external_matched_negatives, metrics
from negative_pool import (dti_random_negatives, evaluation_exclusion, reserved_eval_mask,
                               sample_compound_matched_negatives, sample_negatives, training_exclusion)

KEYS = ["roc_auc", "auprc", "per_target_roc_auc", "precision", "recall", "f1", "mcc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--prop", default="results/gift")
    ap.add_argument("--out", default="results/gift/baseline_models/rf")
    a = ap.parse_args()
    s = a.seed
    data = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    sp = vs.derive_chembl35_splits(data, min_training_pchembl=5.0)
    train, val, test, known = sp["train_pairs"], sp["val_pairs"], sp["test_pairs"], sp["all_positive_pairs"]
    aug, cross, _ = vs.load_external_chembl36_interactions_from_pkls(data)
    reserved = reserved_eval_mask(nc, npr)

    # Training set: identical to both evaluation scripts.
    feat_same, feat_cross = vs.build_pair_feature_fn(data), vs.build_pair_feature_fn(aug)
    rf_neg = sample_negatives(np.random.RandomState(s), np.arange(nc), npr, RF_NEG_RATIO * len(train),
                              training_exclusion(train, reserved))
    X = np.stack([feat_same(c, p) for c, p in train + rf_neg])
    y = np.array([1] * len(train) + [0] * len(rf_neg))
    rf = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1).fit(X, y)

    # Evaluation negatives: same RNG stream as evaluate_chembl35.py.
    excl = evaluation_exclusion(known, reserved)
    rng = np.random.RandomState(vs.SEED)
    sample_negatives(rng, np.arange(nc), npr, 5 * len(test), excl)  # all-compound reference, keeps the stream
    matched = sample_compound_matched_negatives(rng, test, npr, 5, excl)
    random_neg = dti_random_negatives(vs.SEED, train + val + test, npr, 5 * len(test), excl)
    ext_neg = external_matched_negatives(cross, npr, tested_pairs=aug.get("external_tested_pairs", ()))
    benches = {"same_version": (test + random_neg, feat_same, len(test)),
               "same_version_matched": (test + matched, feat_same, len(test)),
               "external": (cross + ext_neg, feat_cross, len(cross))}

    ev = json.load(open(f"{a.prop}/run_{s}/eval_final.json"))["results_by_method"]["random_forest"]
    ex = json.load(open(f"{a.prop}/run_{s}/eval_external_matched.json"))["results"]["random_forest"]
    check, ok = {}, True
    for b, (pairs, feat, npos) in benches.items():
        yb = np.array([1] * npos + [0] * (len(pairs) - npos))
        m = metrics(pairs, yb, rf.predict_proba(np.stack([feat(c, p) for c, p in pairs]))[:, 1])
        ref = ex if b == "external" else ev[b]
        diff = max(abs(m[k] - ref[k]) for k in KEYS if k in m and k in ref)
        ok &= diff < 1e-6  # parallel tree averaging can change the last float digits
        check[b] = {"max_abs_diff_vs_saved": diff, **{k: m[k] for k in KEYS if k in m}}
        print(f"seed {s} {b:22s} max |diff| vs saved = {diff:.2e}")

    os.makedirs(a.out, exist_ok=True)
    joblib.dump(rf, os.path.join(a.out, f"rf_seed_{s}.joblib"), compress=3)
    json.dump({"seed": s, "model": "RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)",
               "features": "888 descriptors per pair (vs.build_pair_feature_fn)", "n_train": int(len(y)),
               "reproduces_saved_results": bool(ok), "benchmarks": check},
              open(os.path.join(a.out, f"rf_seed_{s}.json"), "w"), indent=1)
    print(f"seed {s} saved; reproduces saved results: {ok}")


if __name__ == "__main__":
    main()
