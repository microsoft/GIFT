"""Misclassification analysis of the GIFT model of one run (seed 42 in the paper) on the ChEMBL35 random-negative benchmark (threshold 0.5).

Rebuilds the random-negative test pairs exactly as evaluate_chembl35.py (first draw of the
fixed evaluation RNG), scores them, annotates each pair with label-free properties, and embeds the pairs with
t-SNE in the encoder's pair embedding u_cp = [z_c/|z_c| || z_p/|z_p|] / sqrt(2) (Phase 1 pair embedding).
Writes pairs_random.csv, summary_random.json, and the figure.
"""
import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

import pipeline_utils as vs
from graph_cleaning import clean_graph
from evaluate_chembl35 import load_gift_model
from negative_pool import dti_random_negatives, evaluation_exclusion, reserved_eval_mask

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 8.5, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42})
NAMES_PATH = "data/chembl_names.json"
K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="results/gift/run_42")
    ap.add_argument("--graph_path", default="data/graphs/graph_base.pkl")
    ap.add_argument("--out", default="manuscript/figs/tsne_pairs_predictions")
    args = ap.parse_args()
    torch.set_num_threads(16)
    out_dir = os.path.join(args.run_dir, "misclassification")
    os.makedirs(out_dir, exist_ok=True)

    with open(args.graph_path, "rb") as f:
        data = pickle.load(f)
    graph = clean_graph(data)[0]
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    compound_ids = list(data["compound_ids"])[:nc]
    protein_ids = list(data["protein_ids"])[:npr]
    splits = vs.derive_chembl35_splits(data, min_training_pchembl=5.0)
    train_pairs, test_pairs = splits["train_pairs"], splits["test_pairs"]
    eval_exclude = evaluation_exclusion(splits["all_positive_pairs"], reserved_eval_mask(nc, npr))
    negatives = dti_random_negatives(vs.SEED, train_pairs + splits["val_pairs"] + test_pairs, npr, 5 * len(test_pairs),
                                     eval_exclude)
    pairs = test_pairs + negatives
    y = np.array([1] * len(test_pairs) + [0] * len(negatives))

    device = torch.device("cpu")
    model, _ = load_gift_model(data, device, os.path.join(args.run_dir, "e2e", "trained_model.pth"))
    s = vs.score_gnn_pairs(model, graph, pairs, device)
    with torch.no_grad():
        nt = torch.LongTensor(graph["node_types"])
        _, z = model(torch.FloatTensor(graph["node_features"]), torch.LongTensor(graph["edge_index"]),
                     edge_attr=torch.FloatTensor(graph["edge_attr"]), node_types=nt)
        zc = torch.nn.functional.normalize(z[nt == 0], dim=1).numpy()
        zp = torch.nn.functional.normalize(z[nt == 1], dim=1).numpy()
    U = np.hstack([zc[[c for c, _ in pairs]], zp[[p for _, p in pairs]]]) / np.sqrt(2)

    cdeg = np.bincount([c for c, _ in train_pairs], minlength=nc)
    pdeg = np.bincount([p for _, p in train_pairs], minlength=npr)
    pred = (s >= 0.5).astype(int)
    df = pd.DataFrame({"compound_id": [compound_ids[c] for c, _ in pairs], "protein_id": [protein_ids[p] for _, p in pairs],
                       "y": y, "score": s, "pred": pred,
                       "compound_train_deg": [int(cdeg[c]) for c, _ in pairs],
                       "protein_train_deg": [int(pdeg[p]) for _, p in pairs]})
    df["outcome"] = np.select([(df.y == 1) & (df.pred == 1), (df.y == 1) & (df.pred == 0), (df.y == 0) & (df.pred == 1)],
                              ["TP", "FN", "FP"], "TN")
    meta = vs.load_chembl35_compound_metadata(sorted(set(df.compound_id)))
    tnames = json.load(open(NAMES_PATH))["targets"]
    df["compound_name"] = [((meta.get(c) or {}).get("pref_name") or "") for c in df.compound_id]
    df["protein_name"] = [((tnames.get(p) or {}).get("name") or "") for p in df.protein_id]
    df.to_csv(os.path.join(out_dir, "pairs_random.csv"), index=False)

    idx = NearestNeighbors(n_neighbors=K + 1).fit(U).kneighbors(U, return_distance=False)[:, 1:]
    o = df.outcome.values
    err = df[df.outcome.isin(["FP", "FN"])]
    by_c, by_p = err.compound_id.value_counts(), err.protein_id.value_counts()
    summary = {
        "n": len(df), "counts": df.outcome.value_counts().to_dict(),
        "accuracy": float((df.y == df.pred).mean()), "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred)), "recall": float(recall_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, s)),
        "errors_score_lt_0.1_or_gt_0.9": int(((err.score < 0.1) | (err.score > 0.9)).sum()),
        "errors_score_0.4_0.6": int(((err.score >= 0.4) & (err.score <= 0.6)).sum()),
        "neg_compounds_with_training_interactions": int((df[df.y == 0].compound_train_deg > 0).sum()),
        "fp_compounds_with_training_interactions": int((df[df.outcome == "FP"].compound_train_deg > 0).sum()),
        "fn_protein_deg_0": int(((df.outcome == "FN") & (df.protein_train_deg == 0)).sum()),
        "pos_protein_deg_0": int(((df.y == 1) & (df.protein_train_deg == 0)).sum()),
        "fn_protein_deg_1_5": int(((df.outcome == "FN") & df.protein_train_deg.between(1, 5)).sum()),
        "pos_protein_deg_1_5": int(((df.y == 1) & df.protein_train_deg.between(1, 5)).sum()),
        "fn_protein_deg_gt5": int(((df.outcome == "FN") & (df.protein_train_deg > 5)).sum()),
        "pos_protein_deg_gt5": int(((df.y == 1) & (df.protein_train_deg > 5)).sum()),
        "fn_compound_deg_median": float(df[df.outcome == "FN"].compound_train_deg.median()),
        "tp_compound_deg_median": float(df[df.outcome == "TP"].compound_train_deg.median()),
        "n_compounds": int(df.compound_id.nunique()), "n_compounds_with_error": int(len(by_c)),
        "n_proteins": int(df.protein_id.nunique()), "n_proteins_with_error": int(len(by_p)),
        "max_errors_per_protein": int(by_p.max()),
        "errors": err.sort_values("score")[["compound_id", "compound_name", "protein_id", "protein_name", "outcome",
                                            "score", "compound_train_deg", "protein_train_deg"]].to_dict("records"),
        "top_compounds": [{"id": k, "name": (meta.get(k) or {}).get("pref_name"), "errors": int(v),
                           "fp": int(((err.compound_id == k) & (err.outcome == "FP")).sum()),
                           "fn": int(((err.compound_id == k) & (err.outcome == "FN")).sum()),
                           "n_pos": int(((df.compound_id == k) & (df.y == 1)).sum()),
                           "train_deg": int(df[df.compound_id == k].compound_train_deg.iloc[0])}
                          for k, v in by_c.head(8).items()],
        "knn_outcome_composition": {k: {t: float((o[idx[o == k]] == t).mean()) for t in ("TP", "FN", "FP", "TN")}
                                    for k in ("TP", "FN", "FP", "TN")},
        "knn_same_label_pos": float((y[idx[y == 1]] == 1).mean()), "knn_same_label_neg": float((y[idx[y == 0]] == 0).mean()),
    }
    with open(os.path.join(out_dir, "summary_random.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)

    E = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(U)
    df["tsne1"], df["tsne2"] = E[:, 0], E[:, 1]
    df.to_csv(os.path.join(out_dir, "pairs_random.csv"), index=False)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    styles = [("TN", "#c9c9c9", "o", 6, "True negative"), ("TP", "#1b6ca8", "o", 8, "True positive"),
              ("FP", "#e08a1e", "^", 34, "False positive"), ("FN", "#c0392b", "X", 40, "False negative")]
    for k, col, mk, sz, lab in styles:
        m = o == k
        ax.scatter(E[m, 0], E[m, 1], s=sz, c=col, marker=mk, lw=0.4 if k in ("FP", "FN") else 0,
                   edgecolors="black" if k in ("FP", "FN") else "none", alpha=1 if k in ("FP", "FN") else 0.7,
                   label=f"{lab} (n={m.sum():,})", zorder=3 if k in ("FP", "FN") else 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=300)
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=1, default=str))
    for r in summary["errors"]:
        print(r)


if __name__ == "__main__":
    main()
