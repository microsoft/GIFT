"""Phases 1-2 of GIFT: contrastive pretraining of the edge-aware encoder and soft-edge learning, trained on the clean
graph (training CP edges only). Shared by train_backbone and the backbone controls.
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from model_encoder import (
    TripletContrastiveLoss,
    split_interaction_data,
    analyze_meaningful_validation_categories,
    convert_numpy_to_python,
    set_seed,
    evaluate_with_classifier,
    calculate_balanced_metrics,
)
from model_backbone import GNNBackbone
from graph_cleaning import clean_graph
from negative_pool import reserved_eval_mask

GRAPH_PATH = "data/graphs/graph_similarity_input.pkl"


def build_isolated_edge_index(edge_index, edge_attr, isolate_ids_tensor):
    src_isolated = torch.isin(edge_index[0], isolate_ids_tensor)
    dst_isolated = torch.isin(edge_index[1], isolate_ids_tensor)
    keep_mask = ~(src_isolated | dst_isolated)
    pruned_edge_index = edge_index[:, keep_mask]
    pruned_edge_attr = edge_attr[keep_mask] if edge_attr is not None else None
    return pruned_edge_index, pruned_edge_attr


def find_optimal_threshold_f1_local(y_true: np.ndarray, y_scores: np.ndarray):
    """Simple threshold sweep for best F1, without importing heavy external modules."""
    if y_scores.size == 0:
        return 0.5, 0.0
    thresholds = np.unique(np.round(y_scores, 6))
    if thresholds.size == 0:
        return 0.5, 0.0

    best_thr = 0.5
    best_f1 = -1.0
    y_true_int = y_true.astype(np.int64)
    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(np.int64)
        score = f1_score(y_true_int, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr, best_f1


def train_classifier_phase3(
    model,
    x,
    edge_index,
    edge_attr,
    node_types,
    train_pairs,
    val_pairs,
    data,
    num_epochs=60,
    lr=0.0001,
    device="cuda",
    results_dir="results",
    phase3_pos_weight=1.2,
    seed=42,
):
    """Phase 3: logits-consistent classifier training with hard negatives only."""
    print(f"\n=== PHASE 3: CLASSIFIER TRAINING ({num_epochs} epochs) ===")

    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    model.pair_classifier.train()

    optimizer = torch.optim.Adam(model.pair_classifier.parameters(), lr=lr, weight_decay=0)
    pos_weight_tensor = torch.tensor(float(phase3_pos_weight), dtype=torch.float32, device=device)

    with torch.no_grad():
        _, z = model(x, edge_index, edge_attr, node_types=node_types)
        compound_mask = node_types == 0
        protein_mask = node_types == 1
        z_compounds = z[compound_mask]
        z_proteins = z[protein_mask]

    num_compounds = int(data["num_compounds"])
    num_proteins = int(data["num_proteins"])
    positive_set = set(train_pairs)
    rng = np.random.RandomState(seed)

    def generate_hard_negatives(pairs, num_negatives):
        if num_negatives <= 0:
            return []

        ref_count = min(16, len(pairs))
        if ref_count > 0:
            ref_subset = rng.choice(len(pairs), size=ref_count, replace=False)
            ref_embs = []
            for idx in ref_subset:
                c_idx, p_idx = pairs[int(idx)]
                pair_emb = torch.cat([z_compounds[c_idx], z_proteins[p_idx]], dim=0)
                ref_embs.append(F.normalize(pair_emb, p=2, dim=0))
            ref_emb = torch.stack(ref_embs).mean(dim=0)
        else:
            ref_emb = torch.zeros(z_compounds.size(1) + z_proteins.size(1), device=z_compounds.device)

        candidates = []
        target_candidates = max(num_negatives * 3, 2048)
        for _ in range(target_candidates):
            c_idx = int(rng.randint(0, num_compounds))
            p_idx = int(rng.randint(0, num_proteins))
            if (c_idx, p_idx) in positive_set:
                continue
            pair_emb = torch.cat([z_compounds[c_idx], z_proteins[p_idx]], dim=0)
            pair_emb = F.normalize(pair_emb, p=2, dim=0)
            sim = F.cosine_similarity(pair_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
            candidates.append(((c_idx, p_idx), sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        hard_take = int(num_negatives * 0.75)
        negatives = [p for p, _ in candidates[:hard_take]]

        seen = set(negatives)
        while len(negatives) < num_negatives:
            c_idx = int(rng.randint(0, num_compounds))
            p_idx = int(rng.randint(0, num_proteins))
            pair = (c_idx, p_idx)
            if pair in positive_set or pair in seen:
                continue
            negatives.append(pair)
            seen.add(pair)
        return negatives

    best_val_auprc = -1.0
    patience = 0
    max_patience = 24
    batch_size = 64
    eval_batch_size = 256

    for epoch in range(num_epochs):
        train_negative_pairs = generate_hard_negatives(train_pairs, len(train_pairs))
        val_negative_pairs = generate_hard_negatives(val_pairs, len(val_pairs))

        train_all_pairs = train_pairs + train_negative_pairs
        train_labels = np.array([1] * len(train_pairs) + [0] * len(train_negative_pairs), dtype=np.float32)
        perm = rng.permutation(len(train_all_pairs))
        train_all_pairs = [train_all_pairs[i] for i in perm]
        train_labels = train_labels[perm]

        model.pair_classifier.train()
        total_train_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_all_pairs), batch_size):
            batch_pairs = train_all_pairs[i:i + batch_size]
            batch_labels = torch.tensor(train_labels[i:i + batch_size], dtype=torch.float32, device=device)
            batch_c = torch.tensor([p[0] for p in batch_pairs], dtype=torch.long, device=device)
            batch_p = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits, _ = model.classify_pairs(z_compounds, z_proteins, batch_c, batch_p)
            loss_bce = F.binary_cross_entropy_with_logits(logits, batch_labels, pos_weight=pos_weight_tensor)
            loss_bce.backward()
            torch.nn.utils.clip_grad_norm_(model.pair_classifier.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += float(loss_bce.item())
            num_batches += 1

        model.pair_classifier.eval()
        val_all_pairs = val_pairs + val_negative_pairs
        val_labels_np = np.array([1] * len(val_pairs) + [0] * len(val_negative_pairs), dtype=np.float32)

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(val_all_pairs), eval_batch_size):
                batch_pairs = val_all_pairs[i:i + eval_batch_size]
                batch_c = torch.tensor([p[0] for p in batch_pairs], dtype=torch.long, device=device)
                batch_p = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long, device=device)
                _, probs = model.classify_pairs(z_compounds, z_proteins, batch_c, batch_p)
                all_probs.append(probs.detach().cpu().numpy())

        val_probs = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
        val_auc = roc_auc_score(val_labels_np, val_probs) if val_probs.size > 0 else 0.5
        val_auprc = average_precision_score(val_labels_np, val_probs) if val_probs.size > 0 else 0.0
        val_pred = (val_probs > 0.5).astype(np.int64)
        val_f1 = f1_score(val_labels_np.astype(np.int64), val_pred, zero_division=0)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.pair_classifier.state_dict(), os.path.join(results_dir, "best_classifier.pth"))
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 5 == 0:
            avg_train_loss = total_train_loss / max(1, num_batches)
            print(
                f"  Epoch {epoch+1:3d}/{num_epochs}: train_loss={avg_train_loss:.4f} "
                f"val_auc={val_auc:.4f} val_auprc={val_auprc:.4f} val_f1@0.5={val_f1:.4f}"
            )

        if patience >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    best_classifier_path = os.path.join(results_dir, "best_classifier.pth")
    if os.path.exists(best_classifier_path):
        model.pair_classifier.load_state_dict(torch.load(best_classifier_path, map_location=device))

    return model, 0.5


def main():
    parser = argparse.ArgumentParser(description="GIFT encoder, Phases 1-2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="legacy_results/seed_42")
    parser.add_argument("--edge_dropout_p", type=float, default=0.15)
    parser.add_argument("--gate_hidden_dim", type=int, default=16)
    parser.add_argument("--pair_hidden_dim", type=int, default=288)
    parser.add_argument("--pair_geo_dim", type=int, default=112)
    parser.add_argument("--max_isolation_fraction", type=float, default=0.20)
    parser.add_argument("--max_comp_consistency_weight", type=float, default=0.25)
    parser.add_argument("--max_pair_consistency_weight", type=float, default=0.20)
    parser.add_argument("--pair_consistency_samples", type=int, default=2048)
    parser.add_argument("--max_view_consistency_weight", type=float, default=0.12)
    parser.add_argument("--max_view_pair_consistency_weight", type=float, default=0.08)
    parser.add_argument("--view_consistency_samples", type=int, default=2048)
    parser.add_argument("--num_epochs_phase1", type=int, default=140)
    parser.add_argument("--num_epochs_phase2", type=int, default=140)
    parser.add_argument("--num_epochs_phase3", type=int, default=60)
    parser.add_argument("--phase3_pos_weight", type=float, default=1.2)
    parser.add_argument("--phase2_attention_batch_size", type=int, default=32)
    parser.add_argument("--resume_phase2", action="store_true", help="Skip phase 1 and resume from a saved phase-1 checkpoint")
    parser.add_argument("--phase1_checkpoint", type=str, default="", help="Path to best_model_phase1.pth when using --resume_phase2")
    parser.add_argument("--resume_phase3", action="store_true", help="Skip phases 1/2 and resume from a saved phase-2 checkpoint")
    parser.add_argument("--phase2_checkpoint", type=str, default="", help="Path to best_model_phase2.pth when using --resume_phase3")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--graph_path", type=str, default=GRAPH_PATH)
    args = parser.parse_args()

    print("=" * 80)
    print("GIFT ENCODER: CONTRASTIVE PRETRAINING AND SOFT-EDGE LEARNING")
    print(
        f"seed={args.seed} max_iso_frac={args.max_isolation_fraction} "
        f"max_comp_w={args.max_comp_consistency_weight} max_pair_w={args.max_pair_consistency_weight}"
    )
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    with open(args.graph_path, "rb") as f:
        data = pickle.load(f)
    analyze_meaningful_validation_categories(data)
    clean, clean_train, clean_val, clean_test = clean_graph(data)
    print(f"clean graph: CP edges {data['cp_edge_index'].shape[1]} -> {clean['cp_edge_index'].shape[1]} "
          f"(held-out val+test pairs removed from message passing and edge-mask candidates)")

    num_compounds = int(data["num_compounds"])
    input_dim = data["node_features"].shape[1]
    hidden_dim = 128
    output_dim = 64
    num_layers = 3
    edge_dim = data["edge_attr"].shape[1] if data.get("edge_attr") is not None and data["edge_attr"].size > 0 else 1

    feature_mean = data["node_features"].mean(axis=0)
    feature_std = data["node_features"].std(axis=0)

    model = GNNBackbone(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        edge_dim=edge_dim,
        use_edge_mask=True,
        edge_dropout_p=args.edge_dropout_p,
        gate_hidden_dim=args.gate_hidden_dim,
        pair_hidden_dim=args.pair_hidden_dim,
        pair_geo_dim=args.pair_geo_dim,
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(device)

    x = torch.FloatTensor(clean["node_features"]).to(device)
    edge_index = torch.LongTensor(clean["edge_index"]).to(device)
    edge_attr = torch.FloatTensor(clean["edge_attr"]).to(device)
    node_types = data["node_types"]
    node_types_gpu = torch.from_numpy(node_types).to(device)
    compound_mask = node_types_gpu == 0
    protein_mask = node_types_gpu == 1

    train_pairs, val_pairs, test_pairs = split_interaction_data(data, test_ratio=args.test_ratio, val_ratio=args.val_ratio)
    assert [tuple(map(int, p)) for p in train_pairs] == clean_train

    criterion = TripletContrastiveLoss(
        margin=2.0,
        max_negatives_per_positive=10,
        batch_size=64,
        temperature=1.0,
        pos_weight=1.0,
        neg_weight=1.0,
        hard_negative_mining=True,
    )
    criterion.reserved_mask = reserved_eval_mask(int(data["num_compounds"]), int(data["num_proteins"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, betas=(0.9, 0.999))
    rng = np.random.RandomState(args.seed)

    if args.resume_phase3:
        phase2_ckpt = args.phase2_checkpoint or os.path.join(args.results_dir, "best_model_phase2.pth")
        if not os.path.exists(phase2_ckpt):
            raise FileNotFoundError(f"--resume_phase3 requested but checkpoint not found: {phase2_ckpt}")
        model.load_state_dict(torch.load(phase2_ckpt, map_location=device))
        args.num_epochs_phase1 = 0
        args.num_epochs_phase2 = 0
        print(f"Skipping phases 1 and 2; resuming from phase-2 checkpoint: {phase2_ckpt}")
    elif args.resume_phase2:
        phase1_ckpt = args.phase1_checkpoint or os.path.join(args.results_dir, "best_model_phase1.pth")
        if not os.path.exists(phase1_ckpt):
            raise FileNotFoundError(f"--resume_phase2 requested but checkpoint not found: {phase1_ckpt}")
        model.load_state_dict(torch.load(phase1_ckpt, map_location=device))
        print(f"Skipping phase 1 and resuming from checkpoint: {phase1_ckpt}")
    else:
        print(f"\n=== PHASE 1: TRIPLET + CONSISTENCY ({args.num_epochs_phase1} epochs) ===")
        best_val_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        pairs_by_compound = [[] for _ in range(num_compounds)]
        for idx, (c_idx, p_idx) in enumerate(train_pairs):
            pairs_by_compound[c_idx].append(idx)

        for epoch in range(args.num_epochs_phase1):
            model.train()
            optimizer.zero_grad()

            p = epoch / max(1, args.num_epochs_phase1 - 1)
            iso_frac = args.max_isolation_fraction * p
            comp_w = args.max_comp_consistency_weight * p
            pair_w = args.max_pair_consistency_weight * p
            view_w = args.max_view_consistency_weight * p
            view_pair_w = args.max_view_pair_consistency_weight * p

            num_to_isolate = int(round(iso_frac * num_compounds))

            _, z_full = model(x, edge_index, edge_attr, node_types=node_types_gpu)
            zc_full = z_full[compound_mask]
            zp_full = z_full[protein_mask]
            triplet_loss = criterion(zc_full, zp_full, train_pairs, apply_regularization=False)

            _, z_view2 = model(x, edge_index, edge_attr, node_types=node_types_gpu)
            zc_view2 = z_view2[compound_mask]
            zp_view2 = z_view2[protein_mask]

            view_comp_consistency = 1.0 - F.cosine_similarity(zc_full, zc_view2, dim=1).mean()
            view_prot_consistency = 1.0 - F.cosine_similarity(zp_full, zp_view2, dim=1).mean()
            view_node_consistency = 0.5 * (view_comp_consistency + view_prot_consistency)

            view_pair_consistency = torch.tensor(0.0, device=device)
            if train_pairs:
                if len(train_pairs) > args.view_consistency_samples:
                    sampled_idx = rng.choice(len(train_pairs), size=args.view_consistency_samples, replace=False)
                else:
                    sampled_idx = np.arange(len(train_pairs))
                c_idx = torch.tensor([train_pairs[i][0] for i in sampled_idx], device=device, dtype=torch.long)
                p_idx = torch.tensor([train_pairs[i][1] for i in sampled_idx], device=device, dtype=torch.long)
                pair_full = criterion.create_pair_embedding(zc_full[c_idx], zp_full[p_idx])
                pair_view = criterion.create_pair_embedding(zc_view2[c_idx], zp_view2[p_idx])
                view_pair_consistency = 1.0 - F.cosine_similarity(pair_full, pair_view, dim=1).mean()

            comp_consistency = torch.tensor(0.0, device=device)
            pair_consistency = torch.tensor(0.0, device=device)

            if num_to_isolate > 0:
                isolate_ids = rng.choice(num_compounds, size=num_to_isolate, replace=False)
                isolate_ids_tensor = torch.from_numpy(isolate_ids).long().to(device)

                edge_index_iso, edge_attr_iso = build_isolated_edge_index(edge_index, edge_attr, isolate_ids_tensor)
                _, z_iso = model(x, edge_index_iso, edge_attr_iso, node_types=node_types_gpu)
                zc_iso = z_iso[compound_mask]
                zp_iso = z_iso[protein_mask]

                isolate_idx = torch.from_numpy(isolate_ids).long().to(device)
                comp_consistency = 1.0 - F.cosine_similarity(zc_full[isolate_idx], zc_iso[isolate_idx], dim=1).mean()

                affected_pair_indices = []
                for c in isolate_ids:
                    affected_pair_indices.extend(pairs_by_compound[c])
                if affected_pair_indices:
                    if len(affected_pair_indices) > args.pair_consistency_samples:
                        affected_pair_indices = rng.choice(affected_pair_indices, size=args.pair_consistency_samples, replace=False).tolist()

                    c_idx = torch.tensor([train_pairs[i][0] for i in affected_pair_indices], device=device, dtype=torch.long)
                    p_idx = torch.tensor([train_pairs[i][1] for i in affected_pair_indices], device=device, dtype=torch.long)

                    pair_full = criterion.create_pair_embedding(zc_full[c_idx], zp_full[p_idx])
                    pair_iso = criterion.create_pair_embedding(zc_iso[c_idx], zp_iso[p_idx])
                    pair_consistency = 1.0 - F.cosine_similarity(pair_full, pair_iso, dim=1).mean()

            total_loss = (
                triplet_loss
                + comp_w * comp_consistency
                + pair_w * pair_consistency
                + view_w * view_node_consistency
                + view_pair_w * view_pair_consistency
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                _, z_val = model(x, edge_index, edge_attr, node_types=node_types_gpu)
                zc_val = z_val[compound_mask]
                zp_val = z_val[protein_mask]
                val_loss = criterion(zc_val, zp_val, val_pairs, apply_regularization=False)

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(args.results_dir, "best_model_phase1.pth"))
                else:
                    patience_counter += 1

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1:3d}/{args.num_epochs_phase1}: "
                    f"triplet={triplet_loss.item():.4f} comp={comp_consistency.item():.4f} "
                    f"pair={pair_consistency.item():.4f} view={view_node_consistency.item():.4f} "
                    f"view_pair={view_pair_consistency.item():.4f} val={val_loss.item():.4f}"
                )

            if patience_counter >= 80:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load(os.path.join(args.results_dir, "best_model_phase1.pth")))
                break

        print(f"Phase 1 completed in {(time.time()-start_time)/60:.2f} minutes")

    print(f"\n=== PHASE 2: EDGE MASK LEARNING ({args.num_epochs_phase2} epochs) ===")
    start_time_p2 = time.time()
    data_gpu = {"num_compounds": data["num_compounds"], "num_proteins": data["num_proteins"]}
    if clean["cp_edge_index"].size > 0:
        data_gpu["cp_edge_index"] = torch.LongTensor(clean["cp_edge_index"]).to(device)

    # Force smaller homogeneous edge-mask attention chunks to reduce memory spikes.
    if hasattr(model, "protein_edge_mask"):
        _orig_pp = model.protein_edge_mask.compute_self_attention_scores

        def _pp_with_small_batch(node_embeddings, batch_size=100):
            return _orig_pp(node_embeddings, batch_size=min(batch_size, args.phase2_attention_batch_size))

        model.protein_edge_mask.compute_self_attention_scores = _pp_with_small_batch

    if hasattr(model, "compound_edge_mask"):
        _orig_cc = model.compound_edge_mask.compute_self_attention_scores

        def _cc_with_small_batch(node_embeddings, batch_size=100):
            return _orig_cc(node_embeddings, batch_size=min(batch_size, args.phase2_attention_batch_size))

        model.compound_edge_mask.compute_self_attention_scores = _cc_with_small_batch

    print(f"Using phase-2 homogeneous edge-mask attention batch size: {args.phase2_attention_batch_size}")

    optimizer_p2 = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    best_val_loss_p2 = float("inf")
    patience_counter_p2 = 0

    for epoch in range(args.num_epochs_phase2):
        model.train()
        optimizer_p2.zero_grad()
        data_gpu["epoch"] = epoch

        _, z = model(x, edge_index, edge_attr, node_types=node_types_gpu, apply_edge_mask=True, data=data_gpu)
        zc = z[compound_mask]
        zp = z[protein_mask]
        loss = criterion(zc, zp, train_pairs, apply_regularization=False)
        edge_mask_reg = 1e-4 * sum(p.pow(2).sum() for p in model.edge_mask.parameters())
        total_loss = loss + edge_mask_reg
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_p2.step()

        model.eval()
        with torch.no_grad():
            val_data_gpu = data_gpu.copy()
            val_data_gpu["epoch"] = epoch
            _, z = model(x, edge_index, edge_attr, node_types=node_types_gpu, apply_edge_mask=True, data=val_data_gpu)
            zc = z[compound_mask]
            zp = z[protein_mask]
            val_loss = criterion(zc, zp, val_pairs, apply_regularization=False)

            if val_loss.item() < best_val_loss_p2:
                best_val_loss_p2 = val_loss.item()
                patience_counter_p2 = 0
                torch.save(model.state_dict(), os.path.join(args.results_dir, "best_model_phase2.pth"))
            else:
                patience_counter_p2 += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{args.num_epochs_phase2}: Train={total_loss.item():.4f} Val={val_loss.item():.4f}")

        if patience_counter_p2 >= 200:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(torch.load(os.path.join(args.results_dir, "best_model_phase2.pth")))
            break

    print(f"Phase 2 completed in {(time.time()-start_time_p2)/60:.2f} minutes")
    print(f"backbone saved: {args.results_dir}/best_model_phase2.pth")


if __name__ == "__main__":
    main()
