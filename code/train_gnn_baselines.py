"""Standard GNN link-prediction baselines (GCN, GraphSAGE, GAT) on the paper's benchmarks.

Each baseline follows the standard PyTorch Geometric link-prediction recipe and none of our model's
training procedure:
- 2-layer encoder (888 -> 128 -> 64) on the homogeneous graph, no edge attributes, dot-product decoder;
- supervision edges as in RandomLinkSplit(disjoint_train_ratio=0.3): a fixed 30% of the training
  interactions are supervision targets and are not used for message passing during training;
  at evaluation, all training interactions are used for message passing;
- one negative per positive, resampled every epoch; unweighted binary cross-entropy; Adam, lr 0.01;
- early stopping on validation ROC-AUC (connected graph only).
Shared with every method (data protocol, also used by the random forests): the training graph without
held-out interactions, the 888 node features, the data split, the rule that training negatives exclude
training interactions and the evaluation reserve, the validation set used for checkpoint selection of the
proposed model and for tuning the random forest, and the evaluation pairs and fixed threshold of 0.5.
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, average_precision_score)
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

import pipeline_utils as vs
from graph_cleaning import clean_graph
from evaluate_chembl35 import gnn_validation_negatives
from evaluate_external import external_matched_negatives, per_target_roc_auc
from negative_pool import (dti_random_negatives, evaluation_exclusion, reserved_eval_mask,
                               sample_compound_matched_negatives, sample_negatives,
                               sample_protein_matched_negatives, training_exclusion)


class StdLinkGNN(nn.Module):
    def __init__(self, kind, in_dim, hidden=128, out=64):
        super().__init__()
        if kind == "gcn":
            self.c1, self.c2 = GCNConv(in_dim, hidden), GCNConv(hidden, out)
        elif kind == "sage":
            self.c1, self.c2 = SAGEConv(in_dim, hidden), SAGEConv(hidden, out)
        else:
            self.c1, self.c2 = GATConv(in_dim, hidden // 4, heads=4), GATConv(hidden, out, heads=1)

    def encode(self, x, edge_index):
        return self.c2(F.relu(self.c1(x, edge_index)), edge_index)


def metrics(pairs, y, s):
    pred = (s >= 0.5).astype(int)
    pt, n = per_target_roc_auc(pairs, y, s)
    return {"roc_auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "per_target_roc_auc": pt, "per_target_n": n, "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "mcc": float(matthews_corrcoef(y, pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--models", default="gcn,sage,gat")
    ap.add_argument("--output", required=True)
    ap.add_argument("--graph_path", default="data/graphs/graph_base.pkl")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--disjoint_train_ratio", type=float, default=0.3)
    ap.add_argument("--lead_out", default="", help="also score the 10 driver kinases' lead-prioritization rankings")
    ap.add_argument("--save_models", default="", help="directory to save the trained model state_dicts")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.graph_path, "rb") as f:
        data = pickle.load(f)
    graph, train_all, _, _ = clean_graph(data)
    train_all = sorted({tuple(map(int, p)) for p in train_all})
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])

    # Shared evaluation pairs (identical to evaluate_chembl35 / evaluate_external).
    splits = vs.derive_chembl35_splits(data, min_training_pchembl=5.0)
    test_pairs, val_pairs = splits["test_pairs"], splits["val_pairs"]
    reserved = reserved_eval_mask(nc, npr)
    eval_exclude = evaluation_exclusion(splits["all_positive_pairs"], reserved)
    neg_rng = np.random.RandomState(vs.SEED)
    allc_negs = sample_negatives(neg_rng, np.arange(nc), npr, 5 * len(test_pairs), eval_exclude)
    matched_negs = sample_compound_matched_negatives(neg_rng, test_pairs, npr, 5, eval_exclude)
    random_negs = dti_random_negatives(vs.SEED, splits["train_pairs"] + val_pairs + test_pairs, npr, 5 * len(test_pairs), eval_exclude)
    dti_compounds = sorted({c for c, _ in splits["train_pairs"] + val_pairs + test_pairs})
    pmatched_negs = sample_protein_matched_negatives(np.random.RandomState(vs.SEED), test_pairs, dti_compounds, 5, eval_exclude)
    # Early stopping on the validation set used for the proposed model's checkpoint selection.
    val_negs = gnn_validation_negatives(args.seed, val_pairs, splits["all_positive_pairs"], nc, npr)
    val_all = val_pairs + val_negs
    y_val = np.array([1] * len(val_pairs) + [0] * len(val_negs))
    aug, cross_pairs, _ = vs.load_external_chembl36_interactions_from_pkls(graph)
    ext_negs = external_matched_negatives(cross_pairs, npr, tested_pairs=aug.get("external_tested_pairs", ()))
    # Lead-prioritization driver set, identical to train_dta_baselines.py: the 10 NSCLC driver kinases with
    # >= 5 external actives, ranked over the same external drugs (node indices >= nc in the augmented graph).
    DRIVERS = {"CHEMBL203": "EGFR", "CHEMBL1824": "ERBB2", "CHEMBL4247": "ALK", "CHEMBL3717": "MET",
               "CHEMBL2041": "RET", "CHEMBL5568": "ROS1", "CHEMBL5145": "BRAF",
               "CHEMBL2815": "NTRK1", "CHEMBL4898": "NTRK2", "CHEMBL5608": "NTRK3"}
    pindex = {t: i for i, t in enumerate(list(data["protein_ids"])[:npr])}
    ext_drug_idx = [nc + i for i in range(len(aug["external_compound_ids"]))]
    driver_actives = {}
    for c, p in cross_pairs:
        driver_actives.setdefault(p, set()).add(c)
    lead_drivers = [(pindex[t], name) for t, name in DRIVERS.items()
                    if t in pindex and len(driver_actives.get(pindex[t], set())) >= 5]
    lead_store = {}
    benches = {
        "same_version": (test_pairs + random_negs, np.array([1] * len(test_pairs) + [0] * len(random_negs))),
        "same_version_allcompound": (test_pairs + allc_negs, np.array([1] * len(test_pairs) + [0] * len(allc_negs))),
        "same_version_matched": (test_pairs + matched_negs, np.array([1] * len(test_pairs) + [0] * len(matched_negs))),
        "same_version_protein_matched": (test_pairs + pmatched_negs, np.array([1] * len(test_pairs) + [0] * len(pmatched_negs))),
        "cross_version": (cross_pairs + ext_negs, np.array([1] * len(cross_pairs) + [0] * len(ext_negs))),
    }

    mu = data["node_features"].mean(axis=0)
    sd = data["node_features"].std(axis=0)
    sd[sd == 0] = 1.0

    def tensors(g):
        x = torch.tensor((np.asarray(g["node_features"]) - mu) / sd, dtype=torch.float32, device=device)
        ei = torch.tensor(np.asarray(g["edge_index"]), dtype=torch.long, device=device)
        nt = torch.tensor(np.asarray(g["node_types"]), dtype=torch.long, device=device)
        return x, ei, torch.nonzero(nt == 0).flatten(), torch.nonzero(nt == 1).flatten()

    x, edge_index, cnodes, pnodes = tensors(graph)
    n_nodes = x.size(0)

    def score(model, x_, ei, cn, pn, pairs):
        model.eval()
        with torch.no_grad():
            z = model.encode(x_, ei)
            c = cn[torch.tensor([p[0] for p in pairs], device=device)]
            p = pnodes_of(pn, pairs)
            return torch.sigmoid((z[c] * z[p]).sum(-1)).cpu().numpy()

    def pnodes_of(pn, pairs):
        return pn[torch.tensor([p[1] for p in pairs], device=device)]

    train_exclude = training_exclusion(train_all, reserved)
    results = {}
    for kind in args.models.split(","):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.RandomState(args.seed)
        # Fixed disjoint supervision split (RandomLinkSplit, disjoint_train_ratio).
        perm = rng.permutation(len(train_all))
        n_sup = int(round(args.disjoint_train_ratio * len(train_all)))
        sup = [train_all[i] for i in perm[:n_sup]]
        cn_sup, pn_sup = cnodes[torch.tensor([p[0] for p in sup], device=device)], pnodes_of(pnodes, sup)
        keys = edge_index[0] * n_nodes + edge_index[1]
        drop = torch.cat([cn_sup * n_nodes + pn_sup, pn_sup * n_nodes + cn_sup])
        mp_edges = edge_index[:, ~torch.isin(keys, drop)]

        model = StdLinkGNN(kind, x.size(1)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best, patience, best_state, stop = -1.0, 0, None, args.epochs
        for epoch in range(1, args.epochs + 1):
            negs = sample_negatives(rng, np.arange(nc), npr, len(sup), train_exclude)
            pairs = sup + negs
            y = torch.tensor([1.0] * len(sup) + [0.0] * len(negs), device=device)
            model.train()
            z = model.encode(x, mp_edges)
            c = cnodes[torch.tensor([p[0] for p in pairs], device=device)]
            p = pnodes_of(pnodes, pairs)
            loss = F.binary_cross_entropy_with_logits((z[c] * z[p]).sum(-1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            val_auc = roc_auc_score(y_val, score(model, x, edge_index, cnodes, pnodes, val_all))
            if val_auc > best:
                best, patience = val_auc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.patience:
                    stop = epoch
                    break
        model.load_state_dict(best_state)

        xa, eia, cna, pna = tensors(aug)
        res = {"val_roc_auc": float(best), "stopped_at_epoch": stop}
        for name, (pairs, y) in benches.items():
            if name == "cross_version":
                s = score(model, xa, eia, cna, pna, pairs)
            else:
                s = score(model, x, edge_index, cnodes, pnodes, pairs)
            res[name] = metrics(pairs, y, s)
        results[kind] = res
        if args.save_models:
            os.makedirs(args.save_models, exist_ok=True)
            torch.save({"seed": args.seed, "kind": kind, "val_roc_auc": float(best), "stopped_at_epoch": stop,
                        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()}},
                       os.path.join(args.save_models, f"{kind}_seed_{args.seed}.pth"))
        if args.lead_out:
            dr = {}
            for p_d, name in lead_drivers:
                s = score(model, xa, eia, cna, pna, [(d, p_d) for d in ext_drug_idx])
                order = np.argsort(-s, kind="stable")
                acts = driver_actives.get(p_d, set())
                top10 = [ext_drug_idx[order[j]] for j in range(min(10, len(order)))]
                dr[name] = {"active_at_1": int(ext_drug_idx[order[0]] in acts),
                            "hits_top10": int(sum(d in acts for d in top10)),
                            "n_actives": len(acts & set(ext_drug_idx))}
            lead_store[kind] = {"per_driver": dr,
                                "drivers_active_at_1": int(sum(v["active_at_1"] for v in dr.values())),
                                "hits_top10_sum": int(sum(v["hits_top10"] for v in dr.values())),
                                "n_drivers": len(dr)}
        print(kind, f"stop={stop} val_auc={best:.4f}",
              {b: {k: round(res[b][k], 3) for k in ("roc_auc", "per_target_roc_auc", "mcc")} for b in benches}, flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"seed": args.seed, "recipe": "standard PyG link prediction", "results": results}, f, indent=2)
    if args.lead_out:
        os.makedirs(os.path.dirname(args.lead_out) or ".", exist_ok=True)
        json.dump({"seed": args.seed, "lead": lead_store}, open(args.lead_out, "w"), indent=2)


if __name__ == "__main__":
    main()
