"""Matched GNN comparison (ladder controls A and B): a standard GNN encoder under OUR training protocol and pair head.

Rationale: the published standard GNNs differ from the proposed model in two ways at once --
(1) label exposure: a FIXED 30% disjoint supervision split vs. our ROTATING 30% (all positives are eventually
    supervised, and the supervised edges are removed from message passing every step), and
(2) the decoder: a dot product vs. our fusion pair head (descriptor towers + trust gate + MLP/bilinear scorer).
This script rebuilds the strongest standard encoder (GraphSAGE by default) with BOTH of those matched to the
proposed model, so the only remaining difference from the proposed model is the ENCODER itself
(standard SAGE trained from scratch vs. our contrastive soft-edge encoder). Everything else -- graph, features,
negatives, validation set, evaluation pairs, and threshold -- is identical to the published baselines.

Usage: train_ladder_controls.py --seed S --graph_path G --output OUT [--models sage] [--epochs 300]
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

import pipeline_utils as vs
from graph_cleaning import clean_graph
from evaluate_chembl35 import gnn_validation_negatives
from evaluate_external import external_matched_negatives, per_target_roc_auc
from model_inductive_fusion import InductiveFusionHead
from negative_pool import (dti_random_negatives, evaluation_exclusion, reserved_eval_mask,
                               sample_compound_matched_negatives, sample_negatives,
                               sample_protein_matched_negatives, training_exclusion)


class MatchedGNN(nn.Module):
    """Standard 2-layer GNN encoder (888->128->64) + the proposed model's fusion pair head (or a dot decoder)."""

    def __init__(self, kind, in_dim, feat_dim, hidden=128, out=64, pair_d=192, pair_hidden=384, pair_dropout=0.25,
                 decoder="fusion"):
        super().__init__()
        if kind == "gcn":
            self.c1, self.c2 = GCNConv(in_dim, hidden), GCNConv(hidden, out)
        elif kind == "sage":
            self.c1, self.c2 = SAGEConv(in_dim, hidden), SAGEConv(hidden, out)
        else:
            self.c1, self.c2 = GATConv(in_dim, hidden // 4, heads=4), GATConv(hidden, out, heads=1)
        self.decoder = decoder
        self.head = InductiveFusionHead(out, feat_dim, pair_d, pair_hidden, pair_dropout) if decoder == "fusion" else None

    def encode(self, x, edge_index):
        return self.c2(F.relu(self.c1(x, edge_index)), edge_index)

    def logits(self, z, x_raw, cn, pn, c_idx, p_idx, deg):
        zc, zp = z[cn[c_idx]], z[pn[p_idx]]
        if self.decoder == "dot":
            return (zc * zp).sum(-1)
        xc, xp = x_raw[cn[c_idx]], x_raw[pn[p_idx]]
        conn_c = (deg[cn[c_idx]] > 0).float()
        conn_p = (deg[pn[p_idx]] > 0).float()
        return self.head(zc, zp, xc, xp, conn_c, conn_p)


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
    ap.add_argument("--models", default="sage")
    ap.add_argument("--output", required=True)
    ap.add_argument("--graph_path", default="data/graphs/graph_seed42.pkl")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rotate_ratio", type=float, default=0.3, help="fraction supervised (and removed from MP) each epoch")
    ap.add_argument("--decoder", default="fusion", choices=["fusion", "dot"],
                    help="fusion = our connectivity-gated pair head; dot = plain dot-product decoder")
    ap.add_argument("--save_models", default="", help="directory to save the selected model state_dicts")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.graph_path, "rb") as f:
        data = pickle.load(f)
    graph, train_all, _, _ = clean_graph(data)
    train_all = sorted({tuple(map(int, p)) for p in train_all})
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])

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
    val_negs = gnn_validation_negatives(args.seed, val_pairs, splits["all_positive_pairs"], nc, npr)
    val_all = val_pairs + val_negs
    y_val = np.array([1] * len(val_pairs) + [0] * len(val_negs))
    aug, cross_pairs, _ = vs.load_external_chembl36_interactions_from_pkls(graph)
    ext_negs = external_matched_negatives(cross_pairs, npr, tested_pairs=aug.get("external_tested_pairs", ()))
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
    full_deg = torch.zeros(n_nodes, device=device).index_add_(
        0, edge_index.flatten(), torch.ones(edge_index.numel(), device=device))

    def cidx(pairs):
        return torch.tensor([p[0] for p in pairs], device=device)

    def pidx(pairs):
        return torch.tensor([p[1] for p in pairs], device=device)

    def score(model, x_, ei, cn, pn, pairs):
        model.eval()
        with torch.no_grad():
            z = model.encode(x_, ei)
            deg = torch.zeros(x_.size(0), device=device).index_add_(
                0, ei.flatten(), torch.ones(ei.numel(), device=device))
            return torch.sigmoid(model.logits(z, x_, cn, pn, cidx(pairs), pidx(pairs), deg)).cpu().numpy()

    train_exclude = training_exclusion(train_all, reserved)
    cn_all, pn_all = cnodes[cidx(train_all)], pnodes[pidx(train_all)]
    keys = edge_index[0] * n_nodes + edge_index[1]
    results = {}
    for kind in args.models.split(","):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.RandomState(args.seed)
        model = MatchedGNN(kind, x.size(1), x.size(1), decoder=args.decoder).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best, patience, best_state, stop = -1.0, 0, None, args.epochs
        for epoch in range(1, args.epochs + 1):
            # ROTATING supervision: a fresh 30% of positives each epoch; their edges are removed from message passing.
            perm = rng.permutation(len(train_all))
            n_sup = int(round(args.rotate_ratio * len(train_all)))
            sup_i = perm[:n_sup]
            sup = [train_all[i] for i in sup_i]
            cs, ps = cn_all[sup_i], pn_all[sup_i]
            drop = torch.cat([cs * n_nodes + ps, ps * n_nodes + cs])
            mp_edges = edge_index[:, ~torch.isin(keys, drop)]
            deg = torch.zeros(n_nodes, device=device).index_add_(
                0, mp_edges.flatten(), torch.ones(mp_edges.numel(), device=device))
            negs = sample_negatives(rng, np.arange(nc), npr, len(sup), train_exclude)
            pairs = sup + negs
            y = torch.tensor([1.0] * len(sup) + [0.0] * len(negs), device=device)
            model.train()
            z = model.encode(x, mp_edges)
            logit = model.logits(z, x, cnodes, pnodes, cidx(pairs), pidx(pairs), deg)
            loss = F.binary_cross_entropy_with_logits(logit, y)
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
            s = score(model, xa, eia, cna, pna, pairs) if name == "cross_version" \
                else score(model, x, edge_index, cnodes, pnodes, pairs)
            res[name] = metrics(pairs, y, s)
        results[kind] = res
        if args.save_models:
            os.makedirs(args.save_models, exist_ok=True)
            torch.save({"seed": args.seed, "kind": kind, "decoder": args.decoder, "val_roc_auc": float(best),
                        "stopped_at_epoch": stop, "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()}},
                       os.path.join(args.save_models, f"{kind}_{args.decoder}_seed_{args.seed}.pth"))
        print(kind, f"stop={stop} val_auc={best:.4f}",
              {b: {k: round(res[b][k], 3) for k in ("roc_auc", "per_target_roc_auc", "mcc")} for b in benches}, flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"seed": args.seed, "recipe": f"rotating 30% supervision + {args.decoder} decoder", "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
