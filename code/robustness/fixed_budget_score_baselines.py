"""Per-pair scores of the saved baselines (RF, GCN, GraphSAGE, GAT, DeepDTA, GraphDTA) for point 5 (inference only).

No retraining: loads the saved models of results/gift/baseline_models/{rf,gnn} and results/gift/dta_leadprio/models
and rebuilds each method's inputs exactly as its training/evaluation script did:
  RF            train_rf_baseline.py            (888 pair descriptors, vs.build_pair_feature_fn)
  GCN/SAGE/GAT  train_gnn_baselines.py (standardized features, clean graph; augmented graph for external)
  DeepDTA/GraphDTA train_dta_baselines.py  (SMILES/sequence label encoding, GraphDTA molecular graphs)
Scores the shared validation set (val_pairs + gnn_validation_negatives(seed)), the ChEMBL35 compound-matched test
benchmark and the external benchmark (95 drugs, compound-matched negatives), and checks the ROC-AUC against the saved
metrics of each model. DrugBAN (DGL) is handled by score_drugban_val.py (benchmark scores are saved in
results/gift/baseline_models/drugban/run_s/drugban_dgl/scores.npz).
Writes scores/baselines_seed_<s>.npz.  Usage: python code/robustness/fixed_budget_score_baselines.py --seeds 42 ...
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import json
import os
import pickle
import sys

import joblib
import numpy as np
import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pipeline_utils as vs  # noqa: E402
from graph_cleaning import clean_graph  # noqa: E402
from evaluate_chembl35 import gnn_validation_negatives  # noqa: E402
from evaluate_external import external_matched_negatives  # noqa: E402
from negative_pool import (evaluation_exclusion, reserved_eval_mask, sample_compound_matched_negatives,  # noqa: E402
                               sample_negatives)
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402
from train_gnn_baselines import StdLinkGNN  # noqa: E402
import train_dta_baselines as dta  # noqa: E402

OUT = os.path.join(REPO, "results", "robustness", "descriptor_fixed_budget", "scores")
BENCH = ("val", "same_version_matched", "external")


def one(s, dev):
    data = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    sp = vs.derive_chembl35_splits(data, min_training_pchembl=5.0)
    val_pairs, test_pairs, known = sp["val_pairs"], sp["test_pairs"], sp["all_positive_pairs"]
    excl = evaluation_exclusion(known, reserved_eval_mask(nc, npr))
    rng = np.random.RandomState(vs.SEED)
    sample_negatives(rng, np.arange(nc), npr, 5 * len(test_pairs), excl)
    matched = sample_compound_matched_negatives(rng, test_pairs, npr, 5, excl)
    val_negs = gnn_validation_negatives(s, val_pairs, known, nc, npr)
    aug, cross, _ = vs.load_external_chembl36_interactions_from_pkls(data)
    ext_negs = external_matched_negatives(cross, npr, tested_pairs=aug.get("external_tested_pairs", ()))
    assert int(aug["external_compound_offset"]) == nc
    benches = {"val": (val_pairs, val_negs), "same_version_matched": (test_pairs, matched), "external": (cross, ext_negs)}
    P = {b: pos + neg for b, (pos, neg) in benches.items()}
    Y = {b: np.array([1] * len(pos) + [0] * len(neg)) for b, (pos, neg) in benches.items()}
    out = {}
    for b in BENCH:
        out[f"{b}|pairs"] = np.asarray(P[b], dtype=np.int64)
        out[f"{b}|y"] = Y[b]
    checks = {}

    # Random forest.
    rf = joblib.load(f"results/gift/baseline_models/rf/rf_seed_{s}.joblib")
    rf.set_params(n_jobs=8)
    fs, fc = vs.build_pair_feature_fn(data), vs.build_pair_feature_fn(aug)
    for b in BENCH:
        f = fc if b == "external" else fs
        out[f"rf|{b}|score"] = rf.predict_proba(np.stack([f(c, p) for c, p in P[b]]))[:, 1]
    ref = json.load(open(f"results/gift/baseline_models/rf/rf_seed_{s}.json"))["benchmarks"]
    checks["rf"] = {b: (roc_auc_score(Y[b], out[f"rf|{b}|score"]), ref[b]["roc_auc"])
                    for b in ("same_version_matched", "external")}

    # GCN / GraphSAGE / GAT.
    graph = clean_graph(data)[0]
    gaug, gcross, _ = vs.load_external_chembl36_interactions_from_pkls(graph)
    assert gcross == cross
    mu = data["node_features"].mean(axis=0)
    sd = data["node_features"].std(axis=0)
    sd[sd == 0] = 1.0

    def tensors(g):
        x = torch.tensor((np.asarray(g["node_features"]) - mu) / sd, dtype=torch.float32, device=dev)
        ei = torch.tensor(np.asarray(g["edge_index"]), dtype=torch.long, device=dev)
        nt = torch.tensor(np.asarray(g["node_types"]), dtype=torch.long, device=dev)
        return x, ei, torch.nonzero(nt == 0).flatten(), torch.nonzero(nt == 1).flatten()

    T = {"g": tensors(graph), "a": tensors(gaug)}
    gref = json.load(open(f"results/gift/baseline_models/gnn/gnn_baselines_std_seed_{s}.json"))["results"]
    for kind in ("gcn", "sage", "gat"):
        ck = torch.load(f"results/gift/baseline_models/gnn/{kind}_seed_{s}.pth", map_location=dev, weights_only=False)
        m = StdLinkGNN(kind, T["g"][0].size(1)).to(dev)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        for b in BENCH:
            x, ei, cn, pn = T["a" if b == "external" else "g"]
            with torch.no_grad():
                z = m.encode(x, ei)
                c = cn[torch.tensor([p[0] for p in P[b]], device=dev)]
                p = pn[torch.tensor([p[1] for p in P[b]], device=dev)]
                out[f"{kind}|{b}|score"] = torch.sigmoid((z[c] * z[p]).sum(-1)).cpu().numpy().astype(np.float64)
        checks[kind] = {"val": (roc_auc_score(Y["val"], out[f"{kind}|val|score"]), gref[kind]["val_roc_auc"]),
                        "same_version_matched": (roc_auc_score(Y["same_version_matched"],
                                                               out[f"{kind}|same_version_matched|score"]),
                                                 gref[kind]["same_version_matched"]["roc_auc"]),
                        "external": (roc_auc_score(Y["external"], out[f"{kind}|external|score"]),
                                     gref[kind]["cross_version"]["roc_auc"])}

    # DeepDTA / GraphDTA: inputs exactly as train_dta_baselines.py.
    smi_csv = pd.read_csv("data/chembl_lung_cancer_all_compounds.csv").dropna(subset=["smiles"])
    smi_csv = smi_csv.drop_duplicates("compound_id").set_index("compound_id")["smiles"]
    ext_records = pickle.load(open(aug["external_interactions_pkl"], "rb"))["external_compounds"]
    smiles = [smi_csv[c] for c in list(data["compound_ids"])[:nc]] + \
             [ext_records[c]["smiles"] for c in list(aug["external_compound_ids"])]
    seqs = json.load(open("data/protein_sequences.json"))
    pids = list(data["protein_ids"])[:npr]
    PS = torch.tensor(np.stack([dta.label_seq(seqs[p], dta.CHARPROTSET, dta.MAX_SEQ) for p in pids]), device=dev)
    SS = torch.tensor(np.stack([dta.label_seq(x, dta.CHARISOSMISET, dta.MAX_SMI) for x in smiles]), device=dev)
    graphs = [dta.mol_graph(x) for x in smiles]
    dref = json.load(open(f"results/gift/dta_leadprio/dta_metrics_seed_{s}.json"))["results"]
    for kind in ("deepdta", "graphdta"):
        ck = torch.load(f"results/gift/dta_leadprio/models/{kind}_seed_{s}.pth", map_location=dev, weights_only=False)
        m = dta.build_model(kind, ck["config"]).to(dev)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        for b in BENCH:
            res, bs = [], dta.EVAL_BATCH[kind]
            with torch.no_grad():
                for i in range(0, len(P[b]), bs):
                    pr = P[b][i:i + bs]
                    c = [q[0] for q in pr]
                    seq = PS[torch.tensor([q[1] for q in pr], device=dev)]
                    if kind == "deepdta":
                        lo = m(SS[torch.tensor(c, device=dev)], seq)
                    else:
                        lo = m(Batch.from_data_list([graphs[j] for j in c]).to(dev), seq)
                    res.append(torch.sigmoid(lo).cpu().numpy())
            out[f"{kind}|{b}|score"] = np.concatenate(res).astype(np.float64)
        checks[kind] = {"val": (roc_auc_score(Y["val"], out[f"{kind}|val|score"]), dref[kind]["val_roc_auc"])}
        for b in ("same_version_matched", "external"):
            checks[kind][b] = (roc_auc_score(Y[b], out[f"{kind}|{b}|score"]), dref[kind][b]["roc_auc"])

    worst = max(abs(a - r) for d in checks.values() for a, r in d.values())
    print(f"seed {s} checks (recomputed, saved) ROC-AUC: " + json.dumps(
        {k: {b: [round(a, 5), round(r, 5)] for b, (a, r) in v.items()} for k, v in checks.items()}), flush=True)
    print(f"seed {s} max |ROC-AUC diff| vs saved = {worst:.2e}", flush=True)
    np.savez_compressed(f"{OUT}/baselines_seed_{s}.npz", checks=json.dumps(checks), **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 72)))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    for s in a.seeds:
        one(s, dev)


if __name__ == "__main__":
    main()
