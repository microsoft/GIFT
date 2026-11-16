"""Measured-activity benchmark: can the model tell a MEASURED active from a MEASURED below-threshold compound?

The main benchmarks use unrecorded pairs as negatives, so they mix "is active" with "has been measured".
This benchmark removes that confound: every pair carries a real ChEMBL35 measurement.
  positives : held-out (val+test) pairs with median pChEMBL >= 5          (measured active, never trained on)
  negatives : recorded pairs with median pChEMBL < 5, not training positives (measured below threshold)
Two negative definitions are reported:
  moderate  : median pChEMBL in [3,5)   (assayed, moderate -- the hardest, most defensible contrast)
  all       : median pChEMBL < 5        (every measured-but-not-potent pair)
Held-out actives are removed from the message-passing graph, as in the main evaluation.
Metrics: per-target ROC-AUC (targets with both classes), pooled PR-AUC, and precision/recall/MCC at 0.5.
A random-forest on the same descriptors is reported for contrast. Inference only for the model.
"""

import argparse
import json
import pickle

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

import pipeline_utils as vs
from graph_cleaning import clean_graph
from evaluate_chembl35 import load_gift_model
from negative_pool import reserved_eval_mask, sample_negatives, training_exclusion


def pooled_metrics(y, s):
    yhat = (s >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yhat)) if len(set(yhat)) > 1 else 0.0,
        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
    }


def per_target_roc(pairs, y, s):
    by_t = {}
    for i, (_, p) in enumerate(pairs):
        by_t.setdefault(p, []).append(i)
    aucs = []
    for ix in by_t.values():
        ix = np.array(ix)
        if 0 < y[ix].sum() < len(ix):
            aucs.append(roc_auc_score(y[ix], s[ix]))
    return float(np.mean(aucs)), len(aucs)


def one_seed(seed, graph_path, run_dir, device, do_rf=True):
    with open(graph_path, "rb") as f:
        graph_dict = pickle.load(f)
    model, _ = load_gift_model(graph_dict, device, f"{run_dir}/e2e/trained_model.pth")
    gnn_graph = clean_graph(graph_dict)[0]

    med = {tuple(map(int, k)): float(v) for k, v in dict(graph_dict["all_known_pchembl"]).items()}
    so = graph_dict["split_override"]
    tr = {tuple(map(int, p)) for p in so["train"]}
    held = {tuple(map(int, p)) for p in so["val"]} | {tuple(map(int, p)) for p in so["test"]}

    # Negatives are drawn only from the evaluation reserve, which training negatives never touch, so no benchmark
    # pair is used during training in any role (held-out positives are likewise never trained).
    nc_, npr_ = int(graph_dict["num_compounds"]), int(graph_dict["num_proteins"])
    reserved_mask = reserved_eval_mask(nc_, npr_)

    pos = sorted(p for p in held if med.get(p, 0.0) >= 5.0)
    neg_mod = sorted(p for p in med if 3.0 <= med[p] < 5.0 and p not in tr and reserved_mask[p[0], p[1]])
    neg_all = sorted(p for p in med if med[p] < 5.0 and p not in tr and reserved_mask[p[0], p[1]])

    # Optional RF on the same descriptors (identical recipe to the lead-prioritization baseline).
    rf = feat = None
    if do_rf:
        nc, npr = int(graph_dict["num_compounds"]), int(graph_dict["num_proteins"])
        reserved = reserved_eval_mask(nc, npr)
        train_pairs = sorted(tr)
        rf_negs = sample_negatives(np.random.RandomState(seed), np.arange(nc), npr,
                                   len(train_pairs), training_exclusion(train_pairs, reserved))
        feat = vs.build_pair_feature_fn(graph_dict)
        X = np.stack([feat(c, p) for c, p in train_pairs + rf_negs])
        yv = np.array([1] * len(train_pairs) + [0] * len(rf_negs))
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=32).fit(X, yv)

    out = {}
    for name, neg in (("moderate", neg_mod), ("all", neg_all)):
        pairs = pos + neg
        y = np.array([1] * len(pos) + [0] * len(neg))
        s_gnn = vs.score_gnn_pairs(model, gnn_graph, pairs, device)
        pt, nt = per_target_roc(pairs, y, s_gnn)
        rec = {"gnn": {**pooled_metrics(y, s_gnn), "per_target_roc_auc": pt, "n_targets_both": nt}}
        if do_rf:
            s_rf = rf.predict_proba(np.stack([feat(c, p) for c, p in pairs]))[:, 1]
            prf, _ = per_target_roc(pairs, y, s_rf)
            rec["rf"] = {**pooled_metrics(y, s_rf), "per_target_roc_auc": prf, "n_targets_both": nt}
        out[name] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 72)))
    ap.add_argument("--graph_tmpl", default="data/graphs/graph_seed{s}.pkl")
    ap.add_argument("--run_tmpl", default="results/gift/run_{s}")
    ap.add_argument("--output", default="results/gift/measured_activity.json")
    ap.add_argument("--no_rf", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed = {}
    for s in args.seeds:
        r = one_seed(s, args.graph_tmpl.format(s=s), args.run_tmpl.format(s=s), device, do_rf=not args.no_rf)
        per_seed[s] = r
        g = r["moderate"]["gnn"]
        print(f"seed {s}: [3,5) model per-target ROC-AUC={g['per_target_roc_auc']:.3f} "
              f"PR-AUC={g['pr_auc']:.3f} MCC={g['mcc']:.3f} (pos={g['n_pos']} neg={g['n_neg']} targets={g['n_targets_both']})")

    agg = {}
    for variant in ("moderate", "all"):
        agg[variant] = {}
        for m in (["gnn"] if args.no_rf else ["gnn", "rf"]):
            keys = ["per_target_roc_auc", "pr_auc", "roc_auc", "precision", "recall", "mcc"]
            agg[variant][m] = {k + "_mean": float(np.mean([per_seed[s][variant][m][k] for s in args.seeds])) for k in keys}
            agg[variant][m].update({k + "_std": float(np.std([per_seed[s][variant][m][k] for s in args.seeds])) for k in keys})
    json.dump({"per_seed": per_seed, "agg": agg}, open(args.output, "w"), indent=2)

    print("\n=== aggregate over", len(args.seeds), "seeds ===")
    for variant in ("moderate", "all"):
        print(f"[{variant} negatives]")
        for m in (["gnn"] if args.no_rf else ["gnn", "rf"]):
            a = agg[variant][m]
            print(f"  {m:4s} per-target ROC-AUC {a['per_target_roc_auc_mean']:.3f}  PR-AUC {a['pr_auc_mean']:.3f}  "
                  f"precision {a['precision_mean']:.3f}  recall {a['recall_mean']:.3f}  MCC {a['mcc_mean']:.3f}")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
