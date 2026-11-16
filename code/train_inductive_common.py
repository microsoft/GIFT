"""Shared training utilities for the inductive fusion classifier (data loading, isolation of nodes, validation
scoring), used by train_classifier and train_end_to_end.
"""

import argparse
import contextlib
import io
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from model_encoder import (
    analyze_meaningful_validation_categories,
    calculate_balanced_metrics,
    convert_numpy_to_python,
    evaluate_with_classifier,
    set_seed,
    split_interaction_data,
)
from model_inductive_fusion import GNNInductiveFusion

GRAPH_PATH = "data/graphs/graph_similarity_input.pkl"


def build_isolated_edge_index(edge_index, edge_attr, isolate_node_ids):
    keep = ~(torch.isin(edge_index[0], isolate_node_ids) | torch.isin(edge_index[1], isolate_node_ids))
    return edge_index[:, keep], (edge_attr[keep] if edge_attr is not None else None)


def sample_negatives(rng, compound_pool, num_proteins, n, exclude):
    out, seen = [], set()
    while len(out) < n:
        c = rng.choice(compound_pool, size=2 * (n - len(out)))
        p = rng.randint(0, num_proteins, size=c.size)
        for pair in zip(c.tolist(), p.tolist()):
            if pair not in exclude and pair not in seen:
                out.append(pair)
                seen.add(pair)
                if len(out) == n:
                    break
    return out


def main():
    ap = argparse.ArgumentParser(description="Inductive fusion classifier")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results_dir", type=str, default="legacy_results/seed_42")
    ap.add_argument("--phase2_checkpoint", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--neg_ratio", type=int, default=5)
    ap.add_argument("--iso_fraction", type=float, default=0.35)
    ap.add_argument("--pair_d", type=int, default=192)
    ap.add_argument("--pair_hidden", type=int, default=384)
    ap.add_argument("--pair_dropout", type=float, default=0.25)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
    # Same call sequence as earlier trainers so the train/val/test split is identical.
    with contextlib.redirect_stdout(io.StringIO()):
        analyze_meaningful_validation_categories(data)
        train_pairs, val_pairs, test_pairs = split_interaction_data(data, test_ratio=args.test_ratio, val_ratio=args.val_ratio)
    train_pairs = [tuple(map(int, p)) for p in train_pairs]
    val_pairs = [tuple(map(int, p)) for p in val_pairs]

    num_compounds, num_proteins = int(data["num_compounds"]), int(data["num_proteins"])
    input_dim = data["node_features"].shape[1]
    edge_dim = data["edge_attr"].shape[1]
    model = GNNInductiveFusion(
        input_dim=input_dim, hidden_dim=128, output_dim=64, num_layers=3, edge_dim=edge_dim,
        use_edge_mask=True, edge_dropout_p=0.15, gate_hidden_dim=16,
        feature_mean=data["node_features"].mean(axis=0), feature_std=data["node_features"].std(axis=0),
        pair_d=args.pair_d, pair_hidden=args.pair_hidden, pair_dropout=args.pair_dropout,
    ).to(device)

    backbone = {k: v for k, v in torch.load(args.phase2_checkpoint, map_location=device).items()
                if not k.startswith("pair_classifier.")}
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    assert not unexpected and all(k.startswith("pair_classifier.") for k in missing), (missing, unexpected)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.pair_classifier.parameters():
        p.requires_grad_(True)
    print(f"Loaded backbone from {args.phase2_checkpoint}; head params: "
          f"{sum(p.numel() for p in model.pair_classifier.parameters())}")

    x = torch.FloatTensor(data["node_features"]).to(device)
    edge_index = torch.LongTensor(data["edge_index"]).to(device)
    edge_attr = torch.FloatTensor(data["edge_attr"]).to(device)
    node_types = torch.from_numpy(data["node_types"]).long().to(device)
    compound_mask, protein_mask = node_types == 0, node_types == 1
    compound_node_ids = torch.nonzero(compound_mask).flatten()

    def embed(isolate_compounds):
        """Backbone embeddings with all edges of the given compound indices removed."""
        ei, ea = edge_index, edge_attr
        if len(isolate_compounds) > 0:
            nodes = compound_node_ids[torch.as_tensor(np.asarray(isolate_compounds), dtype=torch.long, device=device)]
            ei, ea = build_isolated_edge_index(edge_index, edge_attr, nodes)
        model.eval()
        with torch.no_grad():
            _, z = model(x, ei, ea, node_types=node_types)
        return z[compound_mask], z[protein_mask]

    rng = np.random.RandomState(args.seed)
    positives = set(train_pairs) | set(val_pairs) | {tuple(map(int, p)) for p in test_pairs}
    all_compounds = np.arange(num_compounds)
    val_compounds = sorted({c for c, _ in val_pairs})
    val_negs = sample_negatives(rng, np.asarray(val_compounds), num_proteins, args.neg_ratio * len(val_pairs), positives)
    val_all = val_pairs + val_negs
    y_val = np.array([1] * len(val_pairs) + [0] * len(val_negs))
    vc = torch.tensor([p[0] for p in val_all], device=device)
    vp = torch.tensor([p[1] for p in val_all], device=device)

    head = model.pair_classifier
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    # Balanced class weighting, matching RF's class_weight="balanced".
    pos_weight = torch.tensor(float(args.neg_ratio), device=device)

    def score_val(isolate):
        zc, zp = embed(isolate)
        head.eval()
        with torch.no_grad():
            _, pr = model.classify_pairs(zc, zp, vc, vp)
        pr = pr.cpu().numpy()
        return roc_auc_score(y_val, pr), average_precision_score(y_val, pr), pr

    best, patience = -1.0, 0
    ckpt_path = os.path.join(args.results_dir, "best_classifier.pth")
    for epoch in range(args.epochs):
        iso = rng.choice(num_compounds, size=int(round(args.iso_fraction * num_compounds)), replace=False)
        zc, zp = embed(iso)
        negs = sample_negatives(rng, all_compounds, num_proteins, args.neg_ratio * len(train_pairs), positives)
        pairs = train_pairs + negs
        y = torch.tensor([1.0] * len(train_pairs) + [0.0] * len(negs), device=device)
        perm = torch.randperm(len(pairs), device=device)
        tc = torch.tensor([p[0] for p in pairs], device=device)[perm]
        tp = torch.tensor([p[1] for p in pairs], device=device)[perm]
        y = y[perm]

        head.train()
        total = 0.0
        for i in range(0, len(pairs), args.batch_size):
            sl = slice(i, i + args.batch_size)
            logits, _ = model.classify_pairs(zc, zp, tc[sl], tp[sl])
            loss = F.binary_cross_entropy_with_logits(logits, y[sl], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            total += loss.item() * logits.numel()
        sched.step()

        auc_c, ap_c, pr_c = score_val([])
        auc_i, ap_i, _ = score_val(val_compounds)
        sel = 0.5 * (ap_c + ap_i)
        if sel > best:
            best, patience = sel, 0
            torch.save(head.state_dict(), ckpt_path)
        else:
            patience += 1
        if (epoch + 1) % 5 == 0 or patience == 0:
            f1c = f1_score(y_val, (pr_c > 0.5).astype(int), zero_division=0)
            print(f"Epoch {epoch+1:3d}: loss={total/len(pairs):.4f} val_auc(conn,iso)=({auc_c:.4f},{auc_i:.4f}) "
                  f"val_ap(conn,iso)=({ap_c:.4f},{ap_i:.4f}) f1@0.5(conn)={f1c:.4f}{' *' if patience == 0 else ''}", flush=True)
        if patience >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    head.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    y_true, y_scores, _ = evaluate_with_classifier(model, x, edge_index, edge_attr, node_types, test_pairs, data, threshold=0.5)
    y_pred = [1 if s > 0.5 else 0 for s in y_scores]
    m = calculate_balanced_metrics(y_true, y_pred, y_scores)
    test_results = {
        "test_accuracy": m["balanced_accuracy"], "test_roc_auc": m["roc_auc"], "test_precision": m["precision"],
        "test_recall": m["recall"], "test_f1": m["f1_score"], "test_mcc": m["mcc"], "best_threshold": 0.5,
    }
    print("Final test results (manuscript protocol):", {k: round(float(v), 4) for k, v in test_results.items()})

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": input_dim, "hidden_dim": 128, "output_dim": 64, "num_layers": 3, "edge_dim": edge_dim,
            "use_edge_mask": True, "edge_dropout_p": 0.15, "gate_hidden_dim": 16,
            "pair_d": args.pair_d, "pair_hidden": args.pair_hidden, "pair_dropout": args.pair_dropout,
            "iso_fraction": args.iso_fraction, "neg_ratio": args.neg_ratio,
            "phase2_checkpoint": args.phase2_checkpoint, "graph_path": GRAPH_PATH,
        },
        "test_results": test_results,
        "training_completed": True,
    }, os.path.join(args.results_dir, "trained_model.pth"))
    with open(os.path.join(args.results_dir, "test_results.json"), "w") as f:
        json.dump(convert_numpy_to_python(test_results), f, indent=2)
    print(f"Model saved to: {args.results_dir}/trained_model.pth")


if __name__ == "__main__":
    main()
