"""
What the soft-edge module learned: compound-protein scores S_cp of the Phase-2 encoder (seed 42) for all
graph pairs, compared with ChEMBL35 held-out interactions (validation + test), which are never in the graph.
Runs on CPU.
"""

import argparse
import json
import os
import pickle

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from graph_cleaning import clean_graph
from model_encoder import split_interaction_data
from model_backbone import GNNBackbone

DRIVERS = {
    "CHEMBL203": "EGFR", "CHEMBL1824": "ERBB2", "CHEMBL4247": "ALK", "CHEMBL3717": "MET",
    "CHEMBL2041": "RET", "CHEMBL5568": "ROS1", "CHEMBL5145": "BRAF",
    "CHEMBL2815": "NTRK1", "CHEMBL4898": "NTRK2", "CHEMBL5608": "NTRK3",
}
FINAL_EPOCH = 199  # last Phase-2 epoch: 1000*(t+1) proposed compound-protein edges


def soft_edge_scores(data, ckpt):
    clean = clean_graph(data)[0]
    model = GNNBackbone(
        input_dim=data["node_features"].shape[1], hidden_dim=128, output_dim=64, num_layers=3,
        edge_dim=data["edge_attr"].shape[1], use_edge_mask=True, edge_dropout_p=0.15,
        gate_hidden_dim=16, pair_hidden_dim=288, pair_geo_dim=112,
        feature_mean=data["node_features"].mean(axis=0), feature_std=data["node_features"].std(axis=0),
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    captured = {}
    model.edge_mask.register_forward_hook(lambda m, inp, out: captured.update(c=inp[0], p=inp[1]))
    nt = torch.from_numpy(np.asarray(data["node_types"]))
    d = {"cp_edge_index": torch.LongTensor(clean["cp_edge_index"]) if "cp_edge_index" in clean else torch.empty((2, 0), dtype=torch.long),
         "num_compounds": int(data["num_compounds"]), "num_proteins": int(data["num_proteins"]), "epoch": FINAL_EPOCH}
    with torch.no_grad():
        model(torch.FloatTensor(clean["node_features"]), torch.LongTensor(clean["edge_index"]),
              edge_attr=torch.FloatTensor(clean["edge_attr"]), node_types=nt, apply_edge_mask=True, data=d)
        return model.edge_mask.compute_attention_scores(captured["c"], captured["p"]).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="results/backbones/backbone_seed_42/best_model_phase2.pth")
    parser.add_argument("--graph_path", default="data/graphs/graph_base.pkl")
    parser.add_argument("--out", default="results/gift/run_42/soft_edge_proposals.json")
    args = parser.parse_args()
    torch.set_num_threads(16)

    data = pickle.load(open(args.graph_path, "rb"))
    S = soft_edge_scores(data, args.backbone)
    C, P = S.shape
    train, val, test = split_interaction_data(data, test_ratio=0.15, val_ratio=0.15)
    train = {tuple(map(int, x)) for x in train}
    held = {tuple(map(int, x)) for x in list(val) + list(test)}
    print("score matrix", S.shape, "train", len(train), "held-out", len(held))

    # Global: share of held-out interactions among the pairs the module proposes at its final epoch.
    k = 1000 * (FINAL_EPOCH + 1)
    cand = np.ones((C, P), dtype=bool)
    for c, p in train:
        cand[c, p] = False
    flat = np.where(cand.ravel())[0]
    order = flat[np.argsort(-S.ravel()[flat], kind="stable")]
    y = np.zeros(C * P, dtype=bool)
    for c, p in held:
        y[c * P + p] = True
    top = order[:k]
    base = y[flat].mean()
    glob = {"k": k, "candidates": int(flat.size), "held_out": int(y.sum()),
            "held_out_in_top_k": int(y[top].sum()), "precision_top_k": float(y[top].mean()),
            "base_rate": float(base), "enrichment": float(y[top].mean() / base),
            "roc_auc_all_candidates": float(roc_auc_score(y[flat], S.ravel()[flat]))}
    # Target-agnostic control: compound mean score plus protein mean score.
    ctrl = (S.mean(1, keepdims=True) + S.mean(0, keepdims=True)).ravel()
    glob["roc_auc_compound_plus_protein_mean"] = float(roc_auc_score(y[flat], ctrl[flat]))
    print(json.dumps(glob, indent=1))

    # Per target: rank graph compounds without a training interaction at the target.
    pids = list(data["protein_ids"])[:P]
    per_target = {}
    for p in range(P):
        mask = np.array([(c, p) not in train for c in range(C)])
        yt = np.array([(c, p) in held for c in range(C)])[mask]
        if yt.sum() < 3:
            continue
        s = S[mask, p]
        per_target[pids[p]] = {"n_held_out": int(yt.sum()), "roc_auc": float(roc_auc_score(yt, s)),
                               "roc_auc_compound_mean": float(roc_auc_score(yt, S.mean(1)[mask]))}
    aucs = np.array([v["roc_auc"] for v in per_target.values()])
    ctl = np.array([v["roc_auc_compound_mean"] for v in per_target.values()])
    summary = {"n_targets": len(per_target), "mean_roc_auc": float(aucs.mean()),
               "mean_roc_auc_compound_mean": float(ctl.mean())}
    print(json.dumps(summary, indent=1))
    for t, g in DRIVERS.items():
        if t in per_target:
            print(g, per_target[t])
    json.dump({"global": glob, "per_target_summary": summary, "per_target": per_target},
              open(args.out, "w"), indent=1)
    np.save(os.path.splitext(args.out)[0] + "_scores.npy", S.astype(np.float32))


if __name__ == "__main__":
    main()
