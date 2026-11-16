"""Evaluate a GIFT model on the held-out ChEMBL 35 benchmarks (random and compound-matched negatives) together with
the random-forest baseline, under the clean protocol: no held-out CP edges in the message-passing graph and a fixed
threshold of 0.5.
"""

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

import pipeline_utils as vs
from cross_version_utils import BASE_GRAPH_PATH
from model_inductive_fusion import GNNInductiveFusion
from graph_cleaning import clean_graph
from negative_pool import (
    dti_random_negatives,
    evaluation_exclusion,
    reserved_eval_mask,
    sample_compound_matched_negatives,
    sample_negatives,
    sample_protein_matched_negatives,
    training_exclusion,
)

RF_GRID = [
    {"n_estimators": 500, "max_depth": md, "min_samples_leaf": leaf, "max_features": mf}
    for md in (None, 30) for leaf in (1, 3) for mf in ("sqrt", 0.2)
]
# One training negative per positive: the standard recipe shared by all baselines (also GCN/GraphSAGE/GAT).
RF_NEG_RATIO = 1


def gnn_validation_negatives(seed, val_pairs, known_pairs, num_compounds, num_proteins, neg_ratio=5):
    """Validation negatives of Phase 3 checkpoint selection (train_classifier/train_end_to_end), drawn from the validation compounds."""
    rng = np.random.RandomState(seed)
    val_compounds = np.asarray(sorted({c for c, _ in val_pairs}))
    exclude = evaluation_exclusion({tuple(map(int, p)) for p in known_pairs}, reserved_eval_mask(num_compounds, num_proteins))
    return sample_negatives(rng, val_compounds, num_proteins, neg_ratio * len(val_pairs), exclude)


def isolate_compounds(graph_data, compound_ids):
    """Copy of the graph with every edge touching `compound_ids` removed."""
    ei = np.asarray(graph_data["edge_index"])
    ids = np.asarray(sorted(compound_ids))
    keep = ~(np.isin(ei[0], ids) | np.isin(ei[1], ids))
    out = dict(graph_data)
    out["edge_index"] = ei[:, keep]
    if graph_data.get("edge_attr") is not None and np.asarray(graph_data["edge_attr"]).size > 0:
        out["edge_attr"] = np.asarray(graph_data["edge_attr"])[keep]
    return out


def load_gift_model(graph_data, device, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("model_config", {})
    model = GNNInductiveFusion(
        input_dim=graph_data["node_features"].shape[1],
        hidden_dim=cfg.get("hidden_dim", 128),
        output_dim=cfg.get("output_dim", 64),
        num_layers=cfg.get("num_layers", 3),
        edge_dim=cfg.get("edge_dim", 2),
        use_edge_mask=cfg.get("use_edge_mask", True),
        edge_dropout_p=cfg.get("edge_dropout_p", 0.15),
        gate_hidden_dim=cfg.get("gate_hidden_dim", 16),
        pair_d=cfg.get("pair_d", 192),
        pair_hidden=cfg.get("pair_hidden", 384),
        pair_dropout=cfg.get("pair_dropout", 0.25),
        descriptor_only=cfg.get("descriptor_only", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="GIFT evaluation on the ChEMBL 35 benchmarks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_checkpoint", type=str, default="legacy_results/seed_42/trained_model.pth")
    parser.add_argument("--min_training_pchembl", type=float, default=5.0)
    parser.add_argument("--negative_ratio", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="legacy_results/seed_42/chembl35_classification_legacy.json")
    parser.add_argument("--graph_path", type=str, default=BASE_GRAPH_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.graph_path, "rb") as f:
        graph_dict = pickle.load(f)
    gift_model, _ = load_gift_model(graph_dict, device, args.model_checkpoint)
    # GNN graph: every held-out (val+test) CP pair removed; splits still come from the original graph.
    gnn_graph = clean_graph(graph_dict)[0]

    num_compounds = int(graph_dict["num_compounds"])
    num_proteins = int(graph_dict["num_proteins"])

    splits = vs.derive_chembl35_splits(graph_dict, min_training_pchembl=args.min_training_pchembl)
    train_pairs = splits["train_pairs"]
    val_pairs = splits["val_pairs"]
    same_version_pairs = splits["test_pairs"]
    known_positive_pairs = splits["all_positive_pairs"]

    v_augmented_data, cross_version_pairs, cross_version_metadata = vs.load_external_chembl36_interactions_from_pkls(graph_dict)
    if not cross_version_pairs:
        raise RuntimeError("No cross-version interactions found")
    gnn_augmented, gnn_cross_pairs, _ = vs.load_external_chembl36_interactions_from_pkls(gnn_graph)
    assert gnn_cross_pairs == cross_version_pairs

    reserved = reserved_eval_mask(num_compounds, num_proteins)
    eval_exclude = evaluation_exclusion(known_positive_pairs, reserved)
    ratio = int(args.negative_ratio)
    all_compounds = np.arange(num_compounds)
    # Fixed seed: identical evaluation negatives for every run.
    neg_rng = np.random.RandomState(vs.SEED)
    # Drawn first to keep the RNG stream of the matched and validation negatives; reported as a reference benchmark.
    allcompound_negatives = sample_negatives(neg_rng, all_compounds, num_proteins, ratio * len(same_version_pairs), eval_exclude)
    matched_negatives = sample_compound_matched_negatives(neg_rng, same_version_pairs, num_proteins, ratio, eval_exclude)
    cross_version_compound_pool = sorted({pair[0] for pair in cross_version_pairs})
    cross_version_negatives = vs.build_external_negative_samples(
        cross_version_compound_pool, num_proteins, cross_version_pairs, negative_ratio=args.negative_ratio,
    )
    val_negatives = sample_negatives(neg_rng, all_compounds, num_proteins, ratio * len(val_pairs), eval_exclude)
    same_version_negatives = dti_random_negatives(
        vs.SEED, train_pairs + val_pairs + same_version_pairs, num_proteins, ratio * len(same_version_pairs), eval_exclude,
    )
    # Separate RNG so the other benchmarks are unchanged; the cold-target benchmark of the protein-disjoint split.
    dti_compounds = sorted({c for c, _ in train_pairs + val_pairs + same_version_pairs})
    pmatched_negatives = sample_protein_matched_negatives(
        np.random.RandomState(vs.SEED), same_version_pairs, dti_compounds, ratio, eval_exclude)
    pmatched_all_pairs = same_version_pairs + pmatched_negatives
    y_true_pmatched = np.array([1] * len(same_version_pairs) + [0] * len(pmatched_negatives), dtype=np.int64)

    same_version_all_pairs = same_version_pairs + same_version_negatives
    allc_all_pairs = same_version_pairs + allcompound_negatives
    matched_all_pairs = same_version_pairs + matched_negatives
    cross_version_all_pairs = cross_version_pairs + cross_version_negatives
    val_all_pairs = val_pairs + val_negatives
    y_true_same = np.array([1] * len(same_version_pairs) + [0] * len(same_version_negatives), dtype=np.int64)
    y_true_allc = np.array([1] * len(same_version_pairs) + [0] * len(allcompound_negatives), dtype=np.int64)
    y_true_matched = np.array([1] * len(same_version_pairs) + [0] * len(matched_negatives), dtype=np.int64)
    y_true_cross = np.array([1] * len(cross_version_pairs) + [0] * len(cross_version_negatives), dtype=np.int64)
    y_true_val = np.array([1] * len(val_pairs) + [0] * len(val_negatives), dtype=np.int64)

    compound_ids = list(graph_dict.get("compound_ids", []))[:num_compounds]
    protein_ids = list(graph_dict.get("protein_ids", []))[:num_proteins]

    def graph_id_lookup(pair):
        return compound_ids[pair[0]], protein_ids[pair[1]]

    external_ids = list(v_augmented_data["external_compound_ids"])
    external_offset = int(v_augmented_data["external_compound_offset"])

    def resolve_compound_id(compound_index):
        if compound_index < num_compounds:
            return compound_ids[compound_index]
        return external_ids[compound_index - external_offset]

    def cross_id_lookup(pair):
        return resolve_compound_id(pair[0]), protein_ids[pair[1]]

    gift_val = vs.score_gnn_pairs(gift_model, gnn_graph, val_all_pairs, device)
    gift_same = vs.score_gnn_pairs(gift_model, gnn_graph, same_version_all_pairs, device)
    gift_allc = vs.score_gnn_pairs(gift_model, gnn_graph, allc_all_pairs, device)
    gift_matched = vs.score_gnn_pairs(gift_model, gnn_graph, matched_all_pairs, device)
    gift_pmatched = vs.score_gnn_pairs(gift_model, gnn_graph, pmatched_all_pairs, device)
    gift_cross = vs.score_gnn_pairs(gift_model, gnn_augmented, cross_version_all_pairs, device)
    same_iso_graph = isolate_compounds(gnn_graph, {p[0] for p in same_version_all_pairs})
    gift_same_iso = vs.score_gnn_pairs(gift_model, same_iso_graph, same_version_all_pairs, device)

    from cross_version_utils import build_tanimoto_scorer
    # Standard nearest-neighbour Tanimoto: same training interactions and 256-bit fingerprints as the models.
    tanimoto_score = build_tanimoto_scorer(graph_dict, cross_version_metadata, train_pairs, 0.0, fp_size=256)
    tanimoto_val = tanimoto_score(val_all_pairs, graph_id_lookup)
    tanimoto_same = tanimoto_score(same_version_all_pairs, graph_id_lookup)
    tanimoto_allc = tanimoto_score(allc_all_pairs, graph_id_lookup)
    tanimoto_matched = tanimoto_score(matched_all_pairs, graph_id_lookup)
    tanimoto_pmatched = tanimoto_score(pmatched_all_pairs, graph_id_lookup)
    tanimoto_cross = tanimoto_score(cross_version_all_pairs, cross_id_lookup)

    from sklearn.ensemble import RandomForestClassifier

    rf_pair_features_same = vs.build_pair_feature_fn(graph_dict)
    rf_pair_features_cross = vs.build_pair_feature_fn(v_augmented_data)
    # Same training-negative policy as the GNN: exclude training positives and the evaluation reserve only.
    rf_train_negatives = sample_negatives(
        np.random.RandomState(args.seed), all_compounds, num_proteins, RF_NEG_RATIO * len(train_pairs),
        training_exclusion(train_pairs, reserved),
    )
    X_rf_train = np.stack([rf_pair_features_same(c, p) for c, p in train_pairs + rf_train_negatives])
    y_rf_train = np.array([1] * len(train_pairs) + [0] * len(rf_train_negatives))
    X_rf_val = np.stack([rf_pair_features_same(c, p) for c, p in val_all_pairs])
    rf_tune_negs = gnn_validation_negatives(args.seed, val_pairs, known_positive_pairs, num_compounds, num_proteins, ratio)
    X_rf_tune = np.stack([rf_pair_features_same(c, p) for c, p in val_pairs + rf_tune_negs])
    y_rf_tune = np.array([1] * len(val_pairs) + [0] * len(rf_tune_negs))
    rf_search = []
    # The control runs (descriptor-only, random augmentation) use only the proposed model's metrics; the RF baseline
    # is fixed from the main runs, so its hyperparameter search is skipped for them.
    rf_grid = RF_GRID[:1] if ("/descriptor_only_seed_" in args.output or "/random_aug_seed_" in args.output) else RF_GRID
    for params in rf_grid:
        clf = RandomForestClassifier(random_state=args.seed, n_jobs=-1, class_weight="balanced", **params)
        clf.fit(X_rf_train, y_rf_train)
        ap = float(average_precision_score(y_rf_tune, clf.predict_proba(X_rf_tune)[:, 1]))
        rf_search.append({"params": params, "val_auprc": ap})
        vs.logger.info(f"RF {params} val_auprc={ap:.4f}")
    rf_best = max(rf_search, key=lambda r: r["val_auprc"])
    rf_clf = RandomForestClassifier(random_state=args.seed, n_jobs=-1, class_weight="balanced", **rf_best["params"])
    rf_clf.fit(X_rf_train, y_rf_train)
    vs.logger.info(f"RF selected on validation: {rf_best}")
    rft_val = rf_clf.predict_proba(X_rf_val)[:, 1]
    rft_same = rf_clf.predict_proba(np.stack([rf_pair_features_same(c, p) for c, p in same_version_all_pairs]))[:, 1]
    rft_allc = rf_clf.predict_proba(np.stack([rf_pair_features_same(c, p) for c, p in allc_all_pairs]))[:, 1]
    rft_matched = rf_clf.predict_proba(np.stack([rf_pair_features_same(c, p) for c, p in matched_all_pairs]))[:, 1]
    rft_pmatched = rf_clf.predict_proba(np.stack([rf_pair_features_same(c, p) for c, p in pmatched_all_pairs]))[:, 1]
    rft_cross = rf_clf.predict_proba(np.stack([rf_pair_features_cross(c, p) for c, p in cross_version_all_pairs]))[:, 1]

    # Primary baseline: scikit-learn default hyperparameters (100 trees, no class weight), no search.
    rf_default = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
    rf_default.fit(X_rf_train, y_rf_train)
    rf_val = rf_default.predict_proba(X_rf_val)[:, 1]
    rf_same = rf_default.predict_proba(np.stack([rf_pair_features_same(c, p) for c, p in same_version_all_pairs]))[:, 1]
    rf_allc = rf_default.predict_proba(np.stack([rf_pair_features_same(c, p) for c, p in allc_all_pairs]))[:, 1]
    rf_matched = rf_default.predict_proba(np.stack([rf_pair_features_same(c, p) for c, p in matched_all_pairs]))[:, 1]
    rf_pmatched = rf_default.predict_proba(np.stack([rf_pair_features_same(c, p) for c, p in pmatched_all_pairs]))[:, 1]
    rf_cross = rf_default.predict_proba(np.stack([rf_pair_features_cross(c, p) for c, p in cross_version_all_pairs]))[:, 1]

    results_by_method = {}
    is_true_probability = {"gift": True, "tanimoto": False, "random_forest": True, "random_forest_tuned": True}

    def per_target_roc_auc(pairs, y_true, scores):
        """Mean ROC-AUC over targets with both classes; target popularity cannot help any method here."""
        by_target = {}
        for i, (_, p) in enumerate(pairs):
            by_target.setdefault(p, []).append(i)
        vals = [roc_auc_score(y_true[ix], scores[ix]) for ix in by_target.values() if 0 < y_true[ix].sum() < len(ix)]
        return float(np.mean(vals)), len(vals)

    def report(name, y_true, scores, threshold, pairs):
        preds = (scores >= threshold).astype(np.int64)
        pt_auc, pt_n = per_target_roc_auc(pairs, y_true, scores)
        return {
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "auprc": float(average_precision_score(y_true, scores)),
            "per_target_roc_auc": pt_auc,
            "per_target_n": pt_n,
            "accuracy": float(accuracy_score(y_true, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, preds)),
            "brier_score": float(brier_score_loss(y_true, scores)) if is_true_probability[name] else None,
            "threshold_used": float(threshold),
        }

    for name, val_scores, same_scores, allc_scores, matched_scores, cross_scores, pmatched_scores in [
        ("gift", gift_val, gift_same, gift_allc, gift_matched, gift_cross, gift_pmatched),
        ("tanimoto", tanimoto_val, tanimoto_same, tanimoto_allc, tanimoto_matched, tanimoto_cross, tanimoto_pmatched),
        ("random_forest", rf_val, rf_same, rf_allc, rf_matched, rf_cross, rf_pmatched),
        ("random_forest_tuned", rft_val, rft_same, rft_allc, rft_matched, rft_cross, rft_pmatched),
    ]:
        threshold = 0.5
        same_metrics = report(name, y_true_same, same_scores, threshold, same_version_all_pairs)
        matched_metrics = report(name, y_true_matched, matched_scores, threshold, matched_all_pairs)
        cross_metrics = report(name, y_true_cross, cross_scores, threshold, cross_version_all_pairs)
        results_by_method[name] = {
            "same_version": same_metrics, "same_version_matched": matched_metrics, "cross_version": cross_metrics,
            "same_version_allcompound": report(name, y_true_allc, allc_scores, threshold, allc_all_pairs),
            "same_version_protein_matched": report(name, y_true_pmatched, pmatched_scores, threshold, pmatched_all_pairs),
        }
        vs.logger.info(
            f"{name} (fixed threshold={threshold:.4f}) | "
            f"same-version: roc_auc={same_metrics['roc_auc']:.4f} auprc={same_metrics['auprc']:.4f} "
            f"f1={same_metrics['f1']:.3f} mcc={same_metrics['mcc']:.3f} || "
            f"matched: roc_auc={matched_metrics['roc_auc']:.4f} auprc={matched_metrics['auprc']:.4f} "
            f"f1={matched_metrics['f1']:.3f} mcc={matched_metrics['mcc']:.3f} || "
            f"cross-version: roc_auc={cross_metrics['roc_auc']:.4f} auprc={cross_metrics['auprc']:.4f} "
            f"f1={cross_metrics['f1']:.3f} mcc={cross_metrics['mcc']:.3f}"
        )

    # Label-free reference scores: number of training interactions of the compound / protein.
    cdeg = np.bincount([c for c, _ in train_pairs], minlength=num_compounds)
    pdeg = np.bincount([p for _, p in train_pairs], minlength=num_proteins)
    label_free_baselines = {}
    for kind in ("compound_train_degree", "protein_train_degree"):
        label_free_baselines[kind] = {}
        for split, pairs, y in [
            ("same_version", same_version_all_pairs, y_true_same),
            ("same_version_allcompound", allc_all_pairs, y_true_allc),
            ("same_version_matched", matched_all_pairs, y_true_matched),
            ("cross_version", cross_version_all_pairs, y_true_cross),
        ]:
            s = np.array([(cdeg[c] if c < num_compounds else 0) if kind.startswith("compound") else pdeg[p]
                          for c, p in pairs], dtype=float)
            label_free_baselines[kind][split] = {
                "roc_auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
                "per_target_roc_auc": per_target_roc_auc(pairs, y, s)[0],
            }
        vs.logger.info(f"label-free {kind}: {label_free_baselines[kind]}")

    same_iso_metrics = report("gift", y_true_same, gift_same_iso, 0.5, same_version_all_pairs)
    results_by_method["gift"]["same_version_isolated"] = same_iso_metrics
    vs.logger.info(
        f"gift same-version with test compounds isolated: roc_auc={same_iso_metrics['roc_auc']:.4f} "
        f"auprc={same_iso_metrics['auprc']:.4f} f1={same_iso_metrics['f1']:.3f} mcc={same_iso_metrics['mcc']:.3f}"
    )

    surpasses_rf_same = all(
        results_by_method["gift"]["same_version"][k] >= results_by_method["random_forest"]["same_version"][k]
        for k in ["roc_auc", "auprc", "f1", "mcc"]
    )
    surpasses_rf_cross = all(
        results_by_method["gift"]["cross_version"][k] >= results_by_method["random_forest"]["cross_version"][k]
        for k in ["roc_auc", "auprc", "f1", "mcc"]
    )

    surpasses_rf_same_iso = all(
        same_iso_metrics[k] >= results_by_method["random_forest"]["same_version"][k]
        for k in ["roc_auc", "auprc", "f1", "mcc"]
    )
    gnn = results_by_method["gift"]
    tuned = results_by_method["random_forest_tuned"]
    surpasses_tuned = {
        split: all(gnn[split][k] >= tuned[split][k] for k in ["roc_auc", "auprc", "f1", "mcc"])
        for split in ["same_version", "same_version_matched", "cross_version"]
    }
    surpasses_rf_matched = all(
        gnn["same_version_matched"][k] >= results_by_method["random_forest"]["same_version_matched"][k]
        for k in ["roc_auc", "auprc", "f1", "mcc"]
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "chembl35_classification",
        "seed": args.seed,
        "threshold_calibration": "none; fixed threshold=0.5 for all methods",
        "gnn_graph": "training CP edges only; all val+test CP pairs removed from message passing and edge-mask candidates",
        "negative_policy": "evaluation negatives drawn from a fixed 20% reserve of pairs (negative_pool); "
                           "same_version: random negatives over compounds with >= 1 known potent interaction (DTI standard); "
                           "same_version_allcompound: random negatives over all compounds (reference); "
                           "training negatives (GNN and RF) exclude training positives and the reserve only",
        "rf_baselines": {
            "random_forest": "primary: scikit-learn default hyperparameters (100 trees, no class weight), "
                             "one training negative per positive, no search",
            "random_forest_tuned": "secondary robustness check: 8-config grid selected on validation AUPRC",
        },
        "rf_selection": {"criterion": "validation AUPRC, GNN checkpoint-selection validation set",
                         "selected": rf_best, "grid": rf_search},
        "num_same_version_positives": len(same_version_pairs),
        "num_same_version_negatives": len(same_version_negatives),
        "num_same_version_matched_negatives": len(matched_negatives),
        "num_cross_version_positives": len(cross_version_pairs),
        "num_cross_version_negatives": len(cross_version_negatives),
        "surpasses_rf_on_roc_auc_auprc_f1_mcc": {
            "same_version": surpasses_rf_same,
            "same_version_isolated": surpasses_rf_same_iso,
            "same_version_matched": surpasses_rf_matched,
            "cross_version": surpasses_rf_cross,
        },
        "surpasses_tuned_rf_on_roc_auc_auprc_f1_mcc": surpasses_tuned,
        "label_free_baselines": label_free_baselines,
        "tanimoto_reference": "raw max Tanimoto (nearest neighbour); all training positives; 256-bit",
        "results_by_method": results_by_method,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    vs.logger.info(f"Saved ChEMBL 35 evaluation to: {args.output}")


if __name__ == "__main__":
    main()
