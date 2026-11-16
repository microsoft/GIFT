"""External ChEMBL36 benchmark with compound-matched negatives.

Every external compound with n_c positives receives 5 n_c negatives at graph proteins where it
has no activity with pChEMBL >= 5, so positives and negatives involve the same compounds, as in
the primary ChEMBL35 benchmark. Scores the model of one run and the default and tuned random
forests of the same run (retrained exactly as in evaluate_chembl35.py).
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

import pipeline_utils as vs
from graph_cleaning import clean_graph
from evaluate_chembl35 import RF_GRID, RF_NEG_RATIO, gnn_validation_negatives, load_gift_model
from negative_pool import reserved_eval_mask, sample_negatives, training_exclusion

MATCHED_SEED = 20260


def external_matched_negatives(cross_pairs, num_proteins, ratio=5, tested_pairs=()):
    """Negatives exclude every tested pair (any ChEMBL36 activity record), not only the positives."""
    pos_set = set(cross_pairs) | set(tested_pairs)
    n_pos = {}
    for c, _ in cross_pairs:
        n_pos[c] = n_pos.get(c, 0) + 1
    rng = np.random.default_rng(MATCHED_SEED)
    negatives = []
    for c in sorted(n_pos):
        candidates = np.array([p for p in range(num_proteins) if (c, p) not in pos_set])
        k = min(ratio * n_pos[c], len(candidates))
        negatives += [(c, int(p)) for p in rng.choice(candidates, size=k, replace=False)]
    return negatives


def per_target_roc_auc(pairs, y, s):
    by_t = {}
    for i, (_, p) in enumerate(pairs):
        by_t.setdefault(p, []).append(i)
    vals = [roc_auc_score(y[ix], s[ix]) for ix in by_t.values() if 0 < y[ix].sum() < len(ix)]
    return float(np.mean(vals)), len(vals)


def metrics(pairs, y, s, thresholded=True):
    pt, n = per_target_roc_auc(pairs, y, s)
    out = {"roc_auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
           "per_target_roc_auc": pt, "per_target_n": n}
    if thresholded:
        pred = (s >= 0.5).astype(int)
        out.update({"precision": float(precision_score(y, pred, zero_division=0)),
                    "recall": float(recall_score(y, pred, zero_division=0)),
                    "f1": float(f1_score(y, pred, zero_division=0)), "mcc": float(matthews_corrcoef(y, pred))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--graph_path", default="data/graphs/graph_base.pkl")
    args = ap.parse_args()

    with open(args.graph_path, "rb") as f:
        data = pickle.load(f)
    gnn_graph = clean_graph(data)[0]
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    splits = vs.derive_chembl35_splits(data, min_training_pchembl=5.0)
    train_pairs, val_pairs, test_pairs = splits["train_pairs"], splits["val_pairs"], splits["test_pairs"]
    aug, cross_pairs, _ = vs.load_external_chembl36_interactions_from_pkls(data)
    gnn_aug, gnn_cross_pairs, _ = vs.load_external_chembl36_interactions_from_pkls(gnn_graph)
    assert gnn_cross_pairs == cross_pairs

    reserved = reserved_eval_mask(nc, npr)
    val_negatives = gnn_validation_negatives(args.seed, val_pairs, splits["all_positive_pairs"], nc, npr)

    negatives = external_matched_negatives(cross_pairs, npr, tested_pairs=aug.get("external_tested_pairs", ()))
    pairs = cross_pairs + negatives
    y = np.array([1] * len(cross_pairs) + [0] * len(negatives))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_gift_model(data, device, args.checkpoint)
    results = {"gift": metrics(pairs, y, vs.score_gnn_pairs(model, gnn_aug, pairs, device))}

    # Control runs (descriptor-only, random augmentation) report only the proposed model; skip the RF baselines,
    # which are taken unchanged from the main runs and otherwise dominate the wall-clock time.
    control_run = "/descriptor_only_seed_" in args.output or "/random_aug_seed_" in args.output
    if not control_run:
        feat_same, feat_cross = vs.build_pair_feature_fn(data), vs.build_pair_feature_fn(aug)
        rf_neg = sample_negatives(np.random.RandomState(args.seed), np.arange(nc), npr, RF_NEG_RATIO * len(train_pairs),
                                  training_exclusion(train_pairs, reserved))
        X_train = np.stack([feat_same(c, p) for c, p in train_pairs + rf_neg])
        y_train = np.array([1] * len(train_pairs) + [0] * len(rf_neg))
        X_val = np.stack([feat_same(c, p) for c, p in val_pairs + val_negatives])
        y_val = np.array([1] * len(val_pairs) + [0] * len(val_negatives))
        X_ext = np.stack([feat_cross(c, p) for c, p in pairs])

        rf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        results["random_forest"] = metrics(pairs, y, rf.predict_proba(X_ext)[:, 1])

        best = None
        for params in RF_GRID:
            clf = RandomForestClassifier(random_state=args.seed, n_jobs=-1, class_weight="balanced", **params)
            clf.fit(X_train, y_train)
            ap_val = float(average_precision_score(y_val, clf.predict_proba(X_val)[:, 1]))
            if best is None or ap_val > best[0]:
                best = (ap_val, params)
        rft = RandomForestClassifier(random_state=args.seed, n_jobs=-1, class_weight="balanced", **best[1])
        rft.fit(X_train, y_train)
        results["random_forest_tuned"] = metrics(pairs, y, rft.predict_proba(X_ext)[:, 1])
        results["random_forest_tuned"]["params"] = best[1]

        pdeg = np.bincount([p for _, p in train_pairs], minlength=npr).astype(float)
        results["compound_train_degree"] = metrics(pairs, y, np.zeros(len(pairs)), thresholded=False)
        results["protein_train_degree"] = metrics(pairs, y, pdeg[[p for _, p in pairs]], thresholded=False)

    out = {"seed": args.seed, "checkpoint": args.checkpoint, "n_positives": len(cross_pairs),
           "n_negatives": len(negatives), "n_compounds": len({c for c, _ in pairs}), "results": results}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    for name, r in results.items():
        print(name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
