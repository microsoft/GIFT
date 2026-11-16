"""Shared helpers for scoring compounds outside the training graph (embedding-space nearest-neighbour score
and Tanimoto similarity) used by pipeline_utils. Not an entry point of the paper's pipeline.
"""
import argparse
import json
import os
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import DataStructs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

import pipeline_utils as vs
from model_pair_base import GNNPairClassifierBase

BASE_GRAPH_PATH = "data/graphs/graph_similarity_input.pkl"


def load_base_arch_model(graph_data, device, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("model_config", {})
    model = GNNPairClassifierBase(
        input_dim=graph_data["node_features"].shape[1],
        hidden_dim=cfg.get("hidden_dim", 128), output_dim=cfg.get("output_dim", 64),
        num_layers=cfg.get("num_layers", 3), edge_dim=cfg.get("edge_dim", 2),
        use_edge_mask=cfg.get("use_edge_mask", True),
        edge_dropout_p=cfg.get("edge_dropout_p", 0.15),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_z_compounds(model, data, device):
    x = torch.FloatTensor(data["node_features"]).to(device)
    edge_index = torch.LongTensor(data["edge_index"]).to(device)
    edge_attr = torch.FloatTensor(data["edge_attr"]).to(device) if data.get("edge_attr") is not None and data["edge_attr"].size > 0 else None
    node_types = torch.LongTensor(data["node_types"]).to(device)
    with torch.no_grad():
        _, z_all = model(x, edge_index, edge_attr=edge_attr, node_types=node_types, apply_edge_mask=False, data=None)
    z_compounds = z_all[node_types == 0].detach()
    z_proteins = z_all[node_types == 1].detach()
    return z_compounds, z_proteins


def build_embedding_nn_scorer(z_compounds_same, z_compounds_cross, train_pairs, pchembl_by_pair, min_training_pchembl):
    target_to_active_indices = defaultdict(list)
    for c_idx, p_idx in train_pairs:
        if pchembl_by_pair.get((c_idx, p_idx), -1e9) >= min_training_pchembl:
            target_to_active_indices[p_idx].append(c_idx)

    z_norm_same = F.normalize(z_compounds_same, dim=1)
    z_norm_cross = F.normalize(z_compounds_cross, dim=1)

    def score(pairs, use_cross: bool):
        z_norm = z_norm_cross if use_cross else z_norm_same
        scores = []
        for c_idx, p_idx in pairs:
            active_indices = target_to_active_indices.get(p_idx)
            if not active_indices:
                scores.append(0.0)
                continue
            active_embeds = z_norm_same[active_indices]  # actives are always original TRAIN compounds
            candidate_embed = z_norm[c_idx]
            sims = torch.matmul(active_embeds, candidate_embed)
            scores.append(float(sims.max().item()))
        return np.asarray(scores, dtype=np.float32)

    return score


def build_tanimoto_scorer(base_data, cross_version_metadata, train_pairs, min_training_pchembl, fp_size=2048):
    infra = vs.build_tanimoto_infrastructure(base_data, train_pairs, min_training_pchembl=min_training_pchembl, fp_size=fp_size)
    fingerprint_by_compound = dict(infra["fingerprint_by_compound"])
    target_to_active_fps = infra["target_to_active_fps"]
    fp_generator = vs.get_morgan_fingerprint_generator(fp_size=fp_size)
    for meta in cross_version_metadata.values():
        compound_id = meta["compound_id"]
        if compound_id not in fingerprint_by_compound and meta.get("smiles"):
            fingerprint = vs.build_rdkit_fingerprint(meta["smiles"], fp_generator)
            if fingerprint is not None:
                fingerprint_by_compound[compound_id] = fingerprint

    def score(pairs, id_lookup):
        scores = []
        for pair in pairs:
            compound_id, protein_id = id_lookup(pair)
            candidate_fp = fingerprint_by_compound.get(compound_id)
            active_fps = target_to_active_fps.get(protein_id)
            if candidate_fp is None or not active_fps:
                scores.append(0.0)
                continue
            scores.append(float(max(DataStructs.BulkTanimotoSimilarity(candidate_fp, active_fps))))
        return np.asarray(scores, dtype=np.float32)

    return score


def main():
    parser = argparse.ArgumentParser(description="Embedding nearest-neighbour + Tanimoto fusion (legacy helper)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy_checkpoint", type=str, default="legacy_results/seed_42/trained_model.pth")
    parser.add_argument("--min_training_pchembl", type=float, default=5.0)
    parser.add_argument("--negative_ratio", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="legacy_results/seed_42/chembl_cross_version_edge_generalization_nn.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(BASE_GRAPH_PATH, "rb") as f:
        graph_dict = pickle.load(f)
    legacy_model = load_base_arch_model(graph_dict, device, args.legacy_checkpoint)

    num_compounds = int(graph_dict["num_compounds"])
    num_proteins = int(graph_dict["num_proteins"])
    compound_ids = list(graph_dict.get("compound_ids", []))[:num_compounds]
    protein_ids = list(graph_dict.get("protein_ids", []))[:num_proteins]

    splits = vs.derive_chembl35_splits(graph_dict, min_training_pchembl=args.min_training_pchembl)
    train_pairs = splits["train_pairs"]
    same_version_pairs = splits["test_pairs"]
    known_positive_pairs = splits["all_positive_pairs"]
    pchembl_by_pair = splits["pchembl_by_pair"]

    legacy_augmented_data, cross_version_pairs, cross_version_metadata = vs.load_external_chembl36_interactions_from_pkls(graph_dict)
    if not cross_version_pairs:
        raise RuntimeError("No cross-version (ChEMBL36-only) interactions found")

    same_version_negatives = vs.build_negative_samples(
        same_version_pairs, num_compounds, num_proteins,
        negative_ratio=args.negative_ratio, excluded_pairs=known_positive_pairs,
    )
    cross_version_compound_pool = sorted({pair[0] for pair in cross_version_pairs})
    cross_version_negatives = vs.build_external_negative_samples(
        cross_version_compound_pool, num_proteins, cross_version_pairs, negative_ratio=args.negative_ratio,
    )
    train_negatives = vs.build_negative_samples(
        train_pairs, num_compounds, num_proteins, negative_ratio=args.negative_ratio,
        excluded_pairs=known_positive_pairs | set(same_version_negatives) | set(cross_version_negatives),
    )

    same_version_all_pairs = same_version_pairs + same_version_negatives
    cross_version_all_pairs = cross_version_pairs + cross_version_negatives
    train_all_pairs = train_pairs + train_negatives
    y_true_same = np.array([1] * len(same_version_pairs) + [0] * len(same_version_negatives), dtype=np.int64)
    y_true_cross = np.array([1] * len(cross_version_pairs) + [0] * len(cross_version_negatives), dtype=np.int64)
    y_true_train = np.array([1] * len(train_pairs) + [0] * len(train_negatives), dtype=np.int64)

    external_ids = list(legacy_augmented_data["external_compound_ids"])
    external_offset = int(legacy_augmented_data["external_compound_offset"])

    def resolve_compound_id(compound_index):
        if compound_index < num_compounds:
            return compound_ids[compound_index]
        return external_ids[compound_index - external_offset]

    def graph_id_lookup(pair):
        return compound_ids[pair[0]], protein_ids[pair[1]]

    def cross_id_lookup(pair):
        return resolve_compound_id(pair[0]), protein_ids[pair[1]]

    # classifier scores (baseline component of the fusion)
    legacy_train = vs.score_gnn_pairs(legacy_model, graph_dict, train_all_pairs, device)
    legacy_same = vs.score_gnn_pairs(legacy_model, graph_dict, same_version_all_pairs, device)
    legacy_cross = vs.score_gnn_pairs(legacy_model, legacy_augmented_data, cross_version_all_pairs, device)

    # embedding-space nearest-neighbor scores 
    z_compounds_same, _ = compute_z_compounds(legacy_model, graph_dict, device)
    z_compounds_cross, _ = compute_z_compounds(legacy_model, legacy_augmented_data, device)
    embedding_nn_score = build_embedding_nn_scorer(
        z_compounds_same, z_compounds_cross, train_pairs, pchembl_by_pair, args.min_training_pchembl,
    )
    nn_train = embedding_nn_score(train_all_pairs, use_cross=False)
    nn_same = embedding_nn_score(same_version_all_pairs, use_cross=False)
    nn_cross = embedding_nn_score(cross_version_all_pairs, use_cross=True)

    # Tanimoto (existing baseline, included in the fusion)
    tanimoto_score = build_tanimoto_scorer(graph_dict, cross_version_metadata, train_pairs, args.min_training_pchembl)
    tanimoto_train = tanimoto_score(train_all_pairs, graph_id_lookup)
    tanimoto_same = tanimoto_score(same_version_all_pairs, graph_id_lookup)
    tanimoto_cross = tanimoto_score(cross_version_all_pairs, cross_id_lookup)

    results_by_method = {}

    def report(name, scores_same, scores_cross):
        roc_auc_same = float(roc_auc_score(y_true_same, scores_same))
        auprc_same = float(average_precision_score(y_true_same, scores_same))
        roc_auc_cross = float(roc_auc_score(y_true_cross, scores_cross))
        auprc_cross = float(average_precision_score(y_true_cross, scores_cross))
        results_by_method[name] = {
            "same_version": {"roc_auc": roc_auc_same, "auprc": auprc_same},
            "cross_version": {"roc_auc": roc_auc_cross, "auprc": auprc_cross},
            "generalization_gap": {"roc_auc": roc_auc_cross - roc_auc_same, "auprc": auprc_cross - auprc_same},
        }
        vs.logger.info(
            f"{name}: same-version ROC-AUC={roc_auc_same:.4f} AUPRC={auprc_same:.4f} | "
            f"cross-version ROC-AUC={roc_auc_cross:.4f} AUPRC={auprc_cross:.4f} | "
            f"gap ROC-AUC={roc_auc_cross - roc_auc_same:+.4f} AUPRC={auprc_cross - auprc_same:+.4f}"
        )

    report("soft_edge_gnn_legacy", legacy_same, legacy_cross)
    report("tanimoto", tanimoto_same, tanimoto_cross)
    report("embedding_nn_legacy", nn_same, nn_cross)

    # 3-feature logistic-regression fusion (classifier + embedding-NN + Tanimoto)
    X_train = np.column_stack([legacy_train, nn_train, tanimoto_train])
    meta_classifier = LogisticRegression(max_iter=1000)
    meta_classifier.fit(X_train, y_true_train)

    embnn_same = meta_classifier.predict_proba(np.column_stack([legacy_same, nn_same, tanimoto_same]))[:, 1]
    embnn_cross = meta_classifier.predict_proba(np.column_stack([legacy_cross, nn_cross, tanimoto_cross]))[:, 1]
    report("embnn_fusion", embnn_same, embnn_cross)
    results_by_method["embnn_fusion"]["meta_classifier_coefficients"] = {
        "legacy_score": float(meta_classifier.coef_[0][0]),
        "embedding_nn_score": float(meta_classifier.coef_[0][1]),
        "tanimoto_score": float(meta_classifier.coef_[0][2]),
        "intercept": float(meta_classifier.intercept_[0]),
    }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "chembl_cross_version_edge_generalization_nn",
        "seed": args.seed,
        "num_same_version_positives": len(same_version_pairs),
        "num_same_version_negatives": len(same_version_negatives),
        "num_cross_version_positives": len(cross_version_pairs),
        "num_cross_version_negatives": len(cross_version_negatives),
        "results_by_method": results_by_method,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    vs.logger.info(f"Saved comparison study to: {args.output}")


if __name__ == "__main__":
    main()
