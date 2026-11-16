"""Per-pair scores of GIFT (main model) and the saved baselines on the three main-table benchmarks of one run.

Reconstructs the evaluation pairs exactly as evaluate_chembl35.py / evaluate_external.py
/ train_dta_baselines.py (same RNG streams), scores them with the frozen saved models (inference only), checks
that the full-benchmark metrics reproduce the saved per-run JSONs, and writes
results/robustness/temporal_direct_target/preds/seed_<s>.npz  (keys '<bench>|pairs', '<bench>|y', '<bench>|<method>', plus ChEMBL ids).
Methods: gift (results/gift/run_s/e2e/trained_model.pth), desc_only (descriptor_only_seed_s), rf (joblib),
gcn/sage/gat (baseline_models/gnn), deepdta/graphdta (dta_leadprio/models), drugban (shared baseline protocol saved scores).
Usage (from repo root): python code/robustness/temporal_predict_pairs.py --seed 42
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import argparse
import contextlib
import io
import json
import os
import pickle
import sys

import joblib
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import pipeline_utils as vs  # noqa: E402
from graph_cleaning import clean_graph  # noqa: E402
from evaluate_chembl35 import load_gift_model  # noqa: E402
from evaluate_external import external_matched_negatives, metrics  # noqa: E402
from negative_pool import (dti_random_negatives, evaluation_exclusion, reserved_eval_mask,  # noqa: E402
                               sample_compound_matched_negatives, sample_negatives)

OUT = "results/robustness/temporal_direct_target/preds"
KEYS = ["roc_auc", "auprc", "precision", "recall", "mcc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--prop", default="results/gift")
    a = ap.parse_args()
    s = a.seed
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
    nc, npr = int(data["num_compounds"]), int(data["num_proteins"])
    with contextlib.redirect_stdout(io.StringIO()):
        graph = clean_graph(data)[0]
        sp = vs.derive_chembl35_splits(data, min_training_pchembl=5.0)
        aug, cross, _ = vs.load_external_chembl36_interactions_from_pkls(data)
        gaug, gcross, _ = vs.load_external_chembl36_interactions_from_pkls(graph)
    assert gcross == cross
    train, val, test, known = sp["train_pairs"], sp["val_pairs"], sp["test_pairs"], sp["all_positive_pairs"]
    reserved = reserved_eval_mask(nc, npr)
    excl = evaluation_exclusion(known, reserved)
    rng = np.random.RandomState(vs.SEED)
    sample_negatives(rng, np.arange(nc), npr, 5 * len(test), excl)  # keeps the RNG stream
    matched = sample_compound_matched_negatives(rng, test, npr, 5, excl)
    random_neg = dti_random_negatives(vs.SEED, train + val + test, npr, 5 * len(test), excl)
    ext_neg = external_matched_negatives(cross, npr, tested_pairs=aug.get("external_tested_pairs", ()))
    benches = {"same_version": (test + random_neg, len(test)),
               "same_version_matched": (test + matched, len(test)),
               "external": (cross + ext_neg, len(cross))}
    ys = {b: np.array([1] * n + [0] * (len(p) - n)) for b, (p, n) in benches.items()}
    scores = {b: {} for b in benches}

    # GIFT and the descriptor-only control (same loader and scorer as the evaluation scripts).
    for name, ck in [("gift", f"{a.prop}/run_{s}/e2e/trained_model.pth"),
                     ("desc_only", f"{a.prop}/descriptor_only_seed_{s}/e2e/trained_model.pth")]:
        model, _ = load_gift_model(data, dev, ck)
        for b, (pairs, _) in benches.items():
            scores[b][name] = vs.score_gnn_pairs(model, gaug if b == "external" else graph, pairs, dev)
        del model
        torch.cuda.empty_cache()

    # Random forest (saved joblib, 888 pair descriptors).
    rf = joblib.load(f"results/gift/baseline_models/rf/rf_seed_{s}.joblib")
    rf.set_params(n_jobs=4)
    fs, fx = vs.build_pair_feature_fn(data), vs.build_pair_feature_fn(aug)
    for b, (pairs, _) in benches.items():
        f = fx if b == "external" else fs
        scores[b]["rf"] = rf.predict_proba(np.stack([f(c, p) for c, p in pairs]))[:, 1]

    # GCN / GraphSAGE / GAT (train_gnn_baselines.py: standardized features, dot-product decoder).
    from train_gnn_baselines import StdLinkGNN
    mu = data["node_features"].mean(axis=0)
    sd = data["node_features"].std(axis=0)
    sd[sd == 0] = 1.0

    def tensors(g):
        x = torch.tensor((np.asarray(g["node_features"]) - mu) / sd, dtype=torch.float32, device=dev)
        ei = torch.tensor(np.asarray(g["edge_index"]), dtype=torch.long, device=dev)
        nt = torch.tensor(np.asarray(g["node_types"]), dtype=torch.long, device=dev)
        return x, ei, torch.nonzero(nt == 0).flatten(), torch.nonzero(nt == 1).flatten()

    tg = {False: tensors(graph), True: tensors(gaug)}
    for kind in ["gcn", "sage", "gat"]:
        m = StdLinkGNN(kind, tg[False][0].size(1)).to(dev)
        m.load_state_dict(torch.load(f"results/gift/baseline_models/gnn/{kind}_seed_{s}.pth",
                                     map_location=dev, weights_only=False)["model_state_dict"])
        m.eval()
        for b, (pairs, _) in benches.items():
            x, ei, cn, pn = tg[b == "external"]
            with torch.no_grad():
                z = m.encode(x, ei)
                c = cn[torch.tensor([p[0] for p in pairs], device=dev)]
                p = pn[torch.tensor([p[1] for p in pairs], device=dev)]
                scores[b][kind] = torch.sigmoid((z[c] * z[p]).sum(-1)).cpu().numpy()

    # DeepDTA / GraphDTA (saved state_dicts, inputs as in train_dta_baselines.py).
    import pandas as pd
    from torch_geometric.data import Batch
    import train_dta_baselines as dta
    smi_csv = pd.read_csv("data/chembl_lung_cancer_all_compounds.csv").dropna(subset=["smiles"])
    smi_csv = smi_csv.drop_duplicates("compound_id").set_index("compound_id")["smiles"]
    ext_records = pickle.load(open(aug["external_interactions_pkl"], "rb"))["external_compounds"]
    ext_ids = list(aug["external_compound_ids"])
    graph_ids = list(data["compound_ids"])[:nc]
    smiles = [smi_csv[c] for c in graph_ids] + [ext_records[c]["smiles"] for c in ext_ids]
    seqs = json.load(open("data/protein_sequences.json"))
    P = torch.tensor(np.stack([dta.label_seq(seqs[p], dta.CHARPROTSET, dta.MAX_SEQ)
                               for p in list(data["protein_ids"])[:npr]]), device=dev)
    S = torch.tensor(np.stack([dta.label_seq(x, dta.CHARISOSMISET, dta.MAX_SMI) for x in smiles]), device=dev)
    needed = sorted({c for pairs, _ in benches.values() for c, _ in pairs})
    graphs = {i: dta.mol_graph(smiles[i]) for i in needed}
    for kind in ["deepdta", "graphdta"]:
        ck = torch.load(f"results/gift/dta_leadprio/models/{kind}_seed_{s}.pth", map_location=dev, weights_only=False)
        m = dta.build_model(kind, ck["config"]).to(dev)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        for b, (pairs, _) in benches.items():
            out = []
            with torch.no_grad():
                for i in range(0, len(pairs), 1024):
                    chunk = pairs[i:i + 1024]
                    c = [q[0] for q in chunk]
                    seq = P[torch.tensor([q[1] for q in chunk], device=dev)]
                    if kind == "deepdta":
                        o = m(S[torch.tensor(c, device=dev)], seq)
                    else:
                        o = m(Batch.from_data_list([graphs[j] for j in c]).to(dev), seq)
                    out.append(torch.sigmoid(o).cpu().numpy())
            scores[b][kind] = np.concatenate(out)
        del m
        torch.cuda.empty_cache()

    # DrugBAN (official DGL runs; per-pair scores saved by the original runs).
    z = np.load(f"results/gift/baseline_models/drugban/run_{s}/dta_baselines_drugban_scores.npz")
    for b, (pairs, _) in benches.items():
        assert np.array_equal(z[f"{b}|pairs"], np.asarray(pairs)) and np.array_equal(z[f"{b}|y"], ys[b])
        scores[b]["drugban"] = z[f"drugban|{b}|score"]

    # DTIAM (frozen pretrained encoders, AutoGluon head per run; per-pair scores saved by results/robustness/dtiam).
    z = np.load(f"results/robustness/dtiam/runs/run_{s}/dta_baselines_dtiam_scores.npz")
    for b, (pairs, _) in benches.items():
        assert np.array_equal(z[f"{b}|pairs"], np.asarray(pairs)) and np.array_equal(z[f"{b}|y"], ys[b])
        scores[b]["dtiam"] = z[f"dtiam|{b}|score"]

    # Reproduction check against the saved metrics.
    ev = json.load(open(f"{a.prop}/run_{s}/eval_final.json"))["results_by_method"]
    ex = json.load(open(f"{a.prop}/run_{s}/eval_external_matched.json"))["results"]
    dsc_ev = json.load(open(f"{a.prop}/descriptor_only_seed_{s}/eval_final.json"))["results_by_method"]
    dsc_ex = json.load(open(f"{a.prop}/descriptor_only_seed_{s}/eval_external_matched.json"))["results"]
    gnn = json.load(open(f"results/gift/baseline_models/gnn/gnn_baselines_std_seed_{s}.json"))["results"]
    dtm = json.load(open(f"results/gift/dta_leadprio/dta_metrics_seed_{s}.json"))["results"]
    ban = json.load(open(f"results/gift/baseline_models/drugban/run_{s}/dta_baselines_drugban.json"))["results"]["drugban"]
    dti = json.load(open(f"results/robustness/dtiam/runs/run_{s}/dta_baselines_dtiam.json"))["results"]["dtiam"]
    check = {}
    for b, (pairs, _) in benches.items():
        gb = "cross_version" if b == "external" else b
        ref = {"gift": ex["gift"] if b == "external" else ev["gift"][b],
               "desc_only": dsc_ex["gift"] if b == "external" else dsc_ev["gift"][b],
               "rf": ex["random_forest"] if b == "external" else ev["random_forest"][b],
               "gcn": gnn["gcn"][gb], "sage": gnn["sage"][gb], "gat": gnn["gat"][gb],
               "deepdta": dtm["deepdta"][b], "graphdta": dtm["graphdta"][b], "drugban": ban[b], "dtiam": dti[b]}
        check[b] = {}
        for meth, sc in scores[b].items():
            mm = metrics(pairs, ys[b], sc)
            check[b][meth] = max(abs(mm[k] - ref[meth][k]) for k in KEYS)
    print(f"seed {s} max |metric - saved|:", json.dumps({b: {k: round(v, 5) for k, v in c.items()} for b, c in check.items()}))

    os.makedirs(OUT, exist_ok=True)
    store = {"compound_ids": np.array(graph_ids + ext_ids), "protein_ids": np.array(list(data["protein_ids"])[:npr]),
             "nc": nc, "ext_offset": int(aug["external_compound_offset"]), "check": json.dumps(check)}
    assert int(aug["external_compound_offset"]) == nc
    for b, (pairs, _) in benches.items():
        store[f"{b}|pairs"] = np.asarray(pairs)
        store[f"{b}|y"] = ys[b]
        for meth, sc in scores[b].items():
            store[f"{b}|{meth}"] = np.asarray(sc, dtype=np.float32)
    np.savez_compressed(f"{OUT}/seed_{s}.npz", **store)


if __name__ == "__main__":
    main()
