"""[pt15 copy of eval_main_fast.py: identical benchmark/lead code; only I/O changed (--ckpt/--out instead of
--run_dir, so the published run directories are read-only) and the repository root added to sys.path.]

Fast GPU-only evaluation of the proposed model for a run directory: the three benchmarks (random negatives,
compound-matched negatives, external ChEMBL~36) and driver-kinase lead counts. No RF/tanimoto baselines.

Usage: python eval_fast_pt15.py --ckpt RUN/e2e/trained_model.pth --seed 42 --out OUT.json
Reproduces the benchmark construction of evaluate_chembl35.py and evaluate_external.py.
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

import pipeline_utils as vs
from graph_cleaning import clean_graph
from evaluate_chembl35 import load_gift_model
from evaluate_external import external_matched_negatives
from negative_pool import (dti_random_negatives, evaluation_exclusion, reserved_eval_mask,
                               sample_compound_matched_negatives, sample_negatives)

DRIVERS = {"CHEMBL203", "CHEMBL1824", "CHEMBL4247", "CHEMBL3717", "CHEMBL2041",
           "CHEMBL5568", "CHEMBL5145", "CHEMBL2815", "CHEMBL4898", "CHEMBL5608"}


def per_target_roc(pairs, y, s):
    by = {}
    for i, (_, p) in enumerate(pairs):
        by.setdefault(p, []).append(i)
    a = [roc_auc_score(y[np.array(ix)], s[np.array(ix)]) for ix in by.values() if 0 < y[np.array(ix)].sum() < len(ix)]
    return float(np.mean(a)), len(a)


def rep(pairs, y, s):
    yh = (s >= 0.5).astype(int)
    pt, n = per_target_roc(pairs, y, s)
    return {"roc_auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "per_target_roc_auc": pt, "per_target_n": n, "precision": float(precision_score(y, yh, zero_division=0)),
            "recall": float(recall_score(y, yh, zero_division=0)), "f1": float(f1_score(y, yh, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, yh)) if len(set(yh)) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--graph", default=None)
    args = ap.parse_args()
    graph = args.graph or f"data/graphs/graph_seed{args.seed}.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(graph, "rb") as f:
        v5 = pickle.load(f)
    gnn_graph = clean_graph(v5)[0]
    model, _ = load_gift_model(v5, device, args.ckpt)

    nc, npr = int(v5["num_compounds"]), int(v5["num_proteins"])
    sp = vs.derive_chembl35_splits(v5, min_training_pchembl=5.0)
    train_pairs, val_pairs, test_pairs = sp["train_pairs"], sp["val_pairs"], sp["test_pairs"]
    known = sp["all_positive_pairs"]
    reserved = reserved_eval_mask(nc, npr)
    excl = evaluation_exclusion(known, reserved)
    ratio = 5
    rng = np.random.RandomState(vs.SEED)
    sample_negatives(rng, np.arange(nc), npr, ratio * len(test_pairs), excl)   # all-compound draw (RNG stream)
    matched_neg = sample_compound_matched_negatives(rng, test_pairs, npr, ratio, excl)
    rand_neg = dti_random_negatives(vs.SEED, train_pairs + val_pairs + test_pairs, npr, ratio * len(test_pairs), excl)

    results = {}
    for name, neg in (("same_version", rand_neg), ("same_version_matched", matched_neg)):
        pairs = test_pairs + neg
        y = np.array([1] * len(test_pairs) + [0] * len(neg))
        results[name] = rep(pairs, y, vs.score_gnn_pairs(model, gnn_graph, pairs, device))

    aug, cross, _ = vs.load_external_chembl36_interactions_from_pkls(v5)
    gnn_aug, gnn_cross, _ = vs.load_external_chembl36_interactions_from_pkls(gnn_graph)
    ext_neg = external_matched_negatives(cross, npr, tested_pairs=aug.get("external_tested_pairs", ()))
    ext_pairs = cross + ext_neg
    y_ext = np.array([1] * len(cross) + [0] * len(ext_neg))
    results["external"] = rep(ext_pairs, y_ext, vs.score_gnn_pairs(model, gnn_aug, ext_pairs, device))

    protein_ids = list(v5["protein_ids"])[:npr]
    pindex = {t: i for i, t in enumerate(protein_ids)}
    ext_idx = sorted({c for c, _ in cross})
    col = {c: i for i, c in enumerate(ext_idx)}
    labels = np.zeros((npr, len(ext_idx)), int)
    for c, p in cross:
        labels[p, col[c]] = 1
    allp = [(c, p) for p in range(npr) for c in ext_idx]
    sc = np.asarray(vs.score_gnn_pairs(model, gnn_aug, allp, device)).reshape(npr, len(ext_idx))
    nact = labels.sum(1)
    t1 = t10 = 0
    drv_aucs = []
    for t in DRIVERS:
        p = pindex.get(t)
        if p is None or not (5 <= nact[p] < len(ext_idx)):
            continue
        order = np.argsort(-sc[p])
        t1 += int(labels[p, order[0]] == 1)
        t10 += int(labels[p, order[:10]].sum())
        drv_aucs.append(roc_auc_score(labels[p], sc[p]))
    lead = {"driver_top1": t1, "driver_top10": t10, "driver_mean_roc_auc": float(np.mean(drv_aucs))}

    # Full lead prioritization: targets with >= 5 known actives among the external drugs, per-target ranking versus
    # the kinase-family and global (target-blind) rankings.
    kin = json.load(open("data/protein_kinase_annotation.json"))
    is_kin = np.array([kin.get(t, {}).get("kinase", False) for t in protein_ids])
    fam_vec, nonkin_vec, glob_vec = sc[is_kin].mean(0), sc[~is_kin].mean(0), sc.mean(0)
    targets = [p for p in range(npr) if 5 <= nact[p] < len(ext_idx)]

    def hits(v, p):
        return int(labels[p, np.argsort(-v)[:10]].sum())

    rows = []
    for p in targets:
        fam = fam_vec if is_kin[p] else nonkin_vec
        rows.append({"p": p, "kin": bool(is_kin[p]), "drv": protein_ids[p] in DRIVERS,
                     "auc": roc_auc_score(labels[p], sc[p]), "fam_auc": roc_auc_score(labels[p], fam),
                     "glob_auc": roc_auc_score(labels[p], glob_vec), "hits": hits(sc[p], p),
                     "fam_hits": hits(fam, p), "rand": 10 * nact[p] / len(ext_idx)})

    def agg(sel):
        r = [x for x in rows if sel(x)]
        return {"n": len(r), "auc": float(np.mean([x["auc"] for x in r])),
                "fam_auc": float(np.mean([x["fam_auc"] for x in r])),
                "glob_auc": float(np.mean([x["glob_auc"] for x in r])),
                "hits": float(np.mean([x["hits"] for x in r])), "fam_hits": float(np.mean([x["fam_hits"] for x in r])),
                "rand_hits": float(np.mean([x["rand"] for x in r])),
                "above_family": int(sum(x["auc"] > x["fam_auc"] for x in r)),
                "auc_gt_05": int(sum(x["auc"] > 0.5 for x in r))}

    lead.update({"all": agg(lambda x: True), "kinase": agg(lambda x: x["kin"]), "drivers": agg(lambda x: x["drv"])})

    json.dump({"seed": args.seed, "results": results, "lead": lead},
              open(args.out, "w"), indent=1)
    r = results
    print(f"seed {args.seed}: random MCC={r['same_version']['mcc']:.3f} rocauc={r['same_version']['roc_auc']:.3f} | "
          f"matched MCC={r['same_version_matched']['mcc']:.3f} | ext MCC={r['external']['mcc']:.3f} "
          f"rocauc={r['external']['roc_auc']:.3f} | driver top1={t1} top10={t10} auc={lead['driver_mean_roc_auc']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
