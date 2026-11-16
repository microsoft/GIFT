"""
Lead prioritization with a GIFT model: rank the 95 external ChEMBL36 compounds for each graph target.

For every target, all 95 external compounds (no edges in the training graph) are ranked by the
model probability, the default random forest (same training data and seed as evaluate_chembl35)
and nearest-neighbour Tanimoto. Ground truth: ChEMBL36 activities with pChEMBL >= 5.
Pre-specified target sets: every target with >= MIN_ACTIVES external actives, and the actionable
NSCLC driver kinases present in the graph with >= MIN_ACTIVES external actives.
Runs on CPU so it does not disturb training jobs on the GPU.
"""

import argparse
import json
import os
import pickle

import numpy as np
import requests
import torch
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

import pipeline_utils as vs
from graph_cleaning import clean_graph
from cross_version_utils import build_tanimoto_scorer
from evaluate_chembl35 import load_gift_model
from negative_pool import reserved_eval_mask, sample_negatives, training_exclusion

MIN_ACTIVES = 5
TOP_N = 10
DRIVERS = {  # actionable NSCLC oncogenic drivers (KRAS is not a graph protein)
    "CHEMBL203": "EGFR", "CHEMBL1824": "ERBB2", "CHEMBL4247": "ALK", "CHEMBL3717": "MET",
    "CHEMBL2041": "RET", "CHEMBL5568": "ROS1", "CHEMBL5145": "BRAF",
    "CHEMBL2815": "NTRK1", "CHEMBL4898": "NTRK2", "CHEMBL5608": "NTRK3",
}
METHODS = ("gnn", "rf", "tanimoto")
KINASE_PATH = "data/protein_kinase_annotation.json"  # UniProt keyword KW-0418 per graph target


def target_metrics(y, s):
    order = np.argsort(-s, kind="stable")
    n_act = int(y.sum())
    hits = int(y[order[:TOP_N]].sum())
    return {
        "n_actives": n_act,
        "roc_auc": float(roc_auc_score(y, s)),
        "hits_top": hits,
        "ef_top": float((hits / TOP_N) / (n_act / len(y))),
        "recovery_curve": np.cumsum(y[order]).tolist(),
    }


CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_DB = os.environ.get("CHEMBL36_DB", "")  # optional; parent names fall back to data/chembl_parent_names.json
PARENT_NAMES = "data/chembl_parent_names.json"


def parent_name(parent_id):
    """Preferred name of the parent molecule (local ChEMBL SQLite, no web access; saved names if the DB is absent)."""
    import sqlite3
    if not os.path.exists(CHEMBL_DB):
        return json.load(open(PARENT_NAMES))["names"].get(parent_id)
    con = sqlite3.connect(f"file:{CHEMBL_DB}?mode=ro", uri=True)
    row = con.execute("SELECT pref_name FROM molecule_dictionary WHERE chembl_id = ?", (parent_id,)).fetchone()
    con.close()
    return row[0] if row else None


def chembl_parent(compound_id):
    """Parent molecule ID and name; activities of salt forms are recorded on the parent."""
    m = requests.get(f"{CHEMBL_API}/molecule/{compound_id}.json", timeout=60).json()
    parent = (m.get("molecule_hierarchy") or {}).get("parent_chembl_id") or compound_id
    if parent != compound_id:
        m = requests.get(f"{CHEMBL_API}/molecule/{parent}.json", timeout=60).json()
    return parent, m.get("pref_name")


def chembl_driver_activities(compound_ids, cache_path):
    """Parent-level ChEMBL web-service activity records of the external compounds at the driver kinases (cached)."""
    if os.path.exists(cache_path):
        return json.load(open(cache_path))
    out = {}
    for cid in compound_ids:
        parent, _ = chembl_parent(cid)
        acts, url = [], f"{CHEMBL_API}/activity.json"
        params = {"parent_molecule_chembl_id": parent, "target_chembl_id__in": ",".join(DRIVERS),
                  "only": "target_chembl_id,pchembl_value", "limit": 1000}
        while url:
            r = requests.get(url, params=params, timeout=120).json()
            acts += r["activities"]
            nxt = r["page_meta"]["next"]
            url, params = ("https://www.ebi.ac.uk" + nxt, None) if nxt else (None, None)
        per_t = {}
        for a in acts:
            d = per_t.setdefault(a["target_chembl_id"], {"n_records": 0, "max_pchembl": None})
            d["n_records"] += 1
            if a.get("pchembl_value"):
                v = float(a["pchembl_value"])
                d["max_pchembl"] = v if d["max_pchembl"] is None else max(d["max_pchembl"], v)
        out[cid] = {"parent": parent, "targets": per_t}
    json.dump(out, open(cache_path, "w"), indent=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", type=str, default="results/gift/run_42")
    parser.add_argument("--graph_path", type=str, default="data/graphs/graph_base.pkl")
    parser.add_argument("--names", type=str, default="data/chembl_names.json")
    parser.add_argument("--top_candidates", type=int, default=5)
    parser.add_argument("--chembl_cache", type=str, default="data/chembl37_driver_activities.json")
    args = parser.parse_args()

    torch.set_num_threads(16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.graph_path, "rb") as f:
        graph_dict = pickle.load(f)
    names = json.load(open(args.names))
    model, _ = load_gift_model(graph_dict, device, os.path.join(args.run_dir, "e2e", "trained_model.pth"))
    gnn_graph = clean_graph(graph_dict)[0]

    num_compounds = int(graph_dict["num_compounds"])
    num_proteins = int(graph_dict["num_proteins"])
    protein_ids = list(graph_dict["protein_ids"])[:num_proteins]
    compound_ids = list(graph_dict["compound_ids"])[:num_compounds]
    splits = vs.derive_chembl35_splits(graph_dict, min_training_pchembl=5.0)
    train_pairs = splits["train_pairs"]

    aug, cross_pairs, cross_meta = vs.load_external_chembl36_interactions_from_pkls(graph_dict)
    gnn_aug, gnn_cross_pairs, _ = vs.load_external_chembl36_interactions_from_pkls(gnn_graph)
    assert gnn_cross_pairs == cross_pairs
    ext_ids = list(aug["external_compound_ids"])
    ext_offset = int(aug["external_compound_offset"])
    ext_idx = sorted({c for c, _ in cross_pairs})
    assert len(ext_idx) == len(ext_ids) == 95

    def resolve(c):
        return compound_ids[c] if c < num_compounds else ext_ids[c - ext_offset]

    def lookup(pair):
        return resolve(pair[0]), protein_ids[pair[1]]

    # Full score matrix: 95 external compounds x all graph proteins.
    all_pairs = [(c, p) for p in range(num_proteins) for c in ext_idx]
    n_ext = len(ext_idx)

    def as_matrix(scores):
        return np.asarray(scores).reshape(num_proteins, n_ext)

    pchembl = {(c, p): cross_meta[(c, p)]["max_pchembl"] for c, p in cross_pairs}
    labels = np.zeros((num_proteins, n_ext), dtype=np.int64)
    col = {c: i for i, c in enumerate(ext_idx)}
    for c, p in cross_pairs:
        labels[p, col[c]] = 1

    scores = {"gnn": as_matrix(vs.score_gnn_pairs(model, gnn_aug, all_pairs, device))}
    tanimoto = build_tanimoto_scorer(graph_dict, cross_meta, train_pairs, 0.0, fp_size=256)
    scores["tanimoto"] = as_matrix(tanimoto(all_pairs, lookup))

    # Default random forest, identical to the primary baseline of the evaluation script.
    ratio = 1
    reserved = reserved_eval_mask(num_compounds, num_proteins)
    rf_negs = sample_negatives(np.random.RandomState(args.seed), np.arange(num_compounds), num_proteins,
                               ratio * len(train_pairs), training_exclusion(train_pairs, reserved))
    feat_same = vs.build_pair_feature_fn(graph_dict)
    feat_cross = vs.build_pair_feature_fn(aug)
    X = np.stack([feat_same(c, p) for c, p in train_pairs + rf_negs])
    y = np.array([1] * len(train_pairs) + [0] * len(rf_negs))
    rf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=64)
    rf.fit(X, y)
    scores["rf"] = as_matrix(rf.predict_proba(np.stack([feat_cross(c, p) for c, p in all_pairs]))[:, 1])

    # Sanity check: per-target ROC-AUC on the external benchmark pairs reproduces eval_final.json.
    negs = vs.build_external_negative_samples(ext_idx, num_proteins, cross_pairs, negative_ratio=5.0)
    bench = cross_pairs + negs
    yb = np.array([1] * len(cross_pairs) + [0] * len(negs))
    by_t = {}
    for i, (_, p) in enumerate(bench):
        by_t.setdefault(p, []).append(i)
    sb = np.array([scores["gnn"][p, col[c]] for c, p in bench])
    pt = np.mean([roc_auc_score(yb[ix], sb[ix]) for ix in by_t.values() if 0 < yb[ix].sum() < len(ix)])
    ref = json.load(open(os.path.join(args.run_dir, "eval_final.json")))
    ref_pt = ref["results_by_method"]["gift"]["cross_version"]["per_target_roc_auc"]
    print(f"sanity: per-target ROC-AUC {pt:.4f} vs eval_final {ref_pt:.4f}")
    assert abs(pt - ref_pt) < 1e-3

    n_act = labels.sum(axis=1)
    eligible = [p for p in range(num_proteins) if MIN_ACTIVES <= n_act[p] < n_ext]
    drivers = [p for p in eligible if protein_ids[p] in DRIVERS]
    # Controls: one compound ranking shared by all targets (mean over all proteins), and one shared within the
    # target's family (kinase vs non-kinase, UniProt keyword Kinase).
    kinase_kw = json.load(open(KINASE_PATH))
    is_kinase = np.array([kinase_kw.get(t, {}).get("kinase", False) for t in protein_ids])
    for m in METHODS:
        s = scores[m]
        scores[m + "_shared"] = np.broadcast_to(s.mean(axis=0), s.shape)
        fam = np.where(is_kinase[:, None], s[is_kinase].mean(axis=0)[None, :], s[~is_kinase].mean(axis=0)[None, :])
        scores[m + "_family"] = fam
    methods = METHODS + tuple(m + suf for m in METHODS for suf in ("_shared", "_family"))
    per_target = {protein_ids[p]: {m: target_metrics(labels[p], scores[m][p]) for m in methods} for p in eligible}

    summary = {"n_kinase_proteins": int(is_kinase.sum()),
               "drivers_all_kinase": bool(all(is_kinase[p] for p in drivers))}
    for name, subset in (("all", eligible), ("kinase", [p for p in eligible if is_kinase[p]]), ("drivers", drivers)):
        summary[name] = {"n_targets": len(subset)}
        for m in methods:
            r = [per_target[protein_ids[p]][m] for p in subset]
            summary[name][m] = {
                "roc_auc_mean": float(np.mean([x["roc_auc"] for x in r])),
                "hits_top_mean": float(np.mean([x["hits_top"] for x in r])),
                "ef_top_mean": float(np.mean([x["ef_top"] for x in r])),
            }
        summary[name]["random_hits_top_mean"] = float(np.mean([TOP_N * n_act[p] / n_ext for p in subset]))
        g = np.array([per_target[protein_ids[p]]["gnn"]["roc_auc"] for p in subset])
        for m in ("rf", "tanimoto"):
            o = np.array([per_target[protein_ids[p]][m]["roc_auc"] for p in subset])
            summary[name][f"gnn_better_than_{m}"] = int((g > o).sum())
            summary[name][f"wilcoxon_p_vs_{m}"] = float(wilcoxon(g, o).pvalue)

    # Selectivity: for compounds active at some but not all drivers, ROC-AUC of their scores across the drivers.
    sel = {}
    for m in METHODS:
        vals = [roc_auc_score(labels[drivers, i], scores[m][drivers, i]) for i in range(n_ext)
                if 0 < labels[drivers, i].sum() < len(drivers)]
        sel[m] = {"mean_roc_auc": float(np.mean(vals)), "n_compounds": len(vals)}
    summary["driver_selectivity"] = sel

    # Sensitivity: parent-level labels (ChEMBL36 set or ChEMBL web pChEMBL >= 5) and only tested compounds as negatives.
    web = chembl_driver_activities([resolve(c) for c in ext_idx], args.chembl_cache)
    sens = {}
    for p in drivers:
        tid = protein_ids[p]
        rec = [web[resolve(c)]["targets"].get(tid) for c in ext_idx]
        act = np.array([labels[p, i] == 1 or (r is not None and (r["max_pchembl"] or 0) >= 5) for i, r in enumerate(rec)])
        tested = np.array([r is not None for r in rec]) | act
        y = act[tested].astype(int)
        row = {"n_tested": int(tested.sum()), "n_actives": int(y.sum()), "n_actives_chembl36": int(labels[p].sum())}
        if 0 < y.sum() < y.size:
            for m in methods:
                row[m] = float(roc_auc_score(y, scores[m][p][tested]))
        sens[DRIVERS[tid]] = row
    summary["driver_sensitivity_tested_only"] = sens
    summary["driver_sensitivity_mean"] = {m: float(np.mean([v[m] for v in sens.values() if m in v])) for m in methods}

    candidates = {}
    for p in drivers:
        tid = protein_ids[p]
        order = np.argsort(-scores["gnn"][p], kind="stable")[: args.top_candidates]
        rows = []
        for i in order:
            c = ext_idx[i]
            cid = resolve(c)
            # Parent and ChEMBL 37 driver-kinase records from the saved snapshot (no web access during runs).
            parent = web[cid]["parent"]
            row = {"compound_id": cid, "name": names["compounds"][cid]["name"],
                   "parent_id": parent, "parent_name": parent_name(parent),
                   "gnn_prob": float(scores["gnn"][p, i]),
                   "rank_rf": int((scores["rf"][p] > scores["rf"][p, i]).sum() + 1),
                   "rank_tanimoto": int((scores["tanimoto"][p] > scores["tanimoto"][p, i]).sum() + 1),
                   "chembl36_pchembl": pchembl.get((c, p))}
            if row["chembl36_pchembl"] is None:
                row["chembl_web"] = web[cid]["targets"].get(tid, {"n_records": 0, "max_pchembl": None})
            rows.append(row)
        candidates[tid] = {"gene": DRIVERS[tid], "rows": rows}

    out = {
        "seed": args.seed, "run_dir": args.run_dir, "min_actives": MIN_ACTIVES, "top_n": TOP_N,
        "n_external_compounds": n_ext, "summary": summary,
        "driver_targets": {protein_ids[p]: DRIVERS[protein_ids[p]] for p in drivers},
        "per_target": per_target, "top_candidates": candidates,
        "driver_scores": {protein_ids[p]: {resolve(c): float(scores["gnn"][p, i]) for i, c in enumerate(ext_idx)}
                          for p in drivers},
        "driver_labels": {protein_ids[p]: [resolve(c) for i, c in enumerate(ext_idx) if labels[p, i]] for p in drivers},
        "sanity_per_target_roc_auc": {"recomputed": float(pt), "eval_final": float(ref_pt)},
    }
    path = os.path.join(args.run_dir, "lead_prioritization.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for tid, v in candidates.items():
        print(tid, v["gene"])
        for r in v["rows"]:
            print("  ", r)
    print("saved", path)


if __name__ == "__main__":
    main()
