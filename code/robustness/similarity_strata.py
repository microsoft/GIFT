"""Point 13: external ROC-AUC/PR-AUC by compound-to-training similarity tertile, per seed (42-71), for GIFT
(the main model, GIFT_main) and the baselines whose models or scores are saved (RF, DeepDTA, GraphDTA, DrugBAN, DTIAM).

Stratification is identical to analyze_external_generalization.py: per seed, each external compound gets its max
ECFP4 (r=2, 2048 bit) Tanimoto similarity to the compounds of that seed's training interactions; tertile edges are the
1/3 and 2/3 quantiles over the benchmark *pairs* (pair-weighted), low <= e1 < mid <= e2 < high.
CPU only. Usage: python pt13_similarity_strata.py --seed 42  (writes pt13_out/seed_42.npz/json)
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = _REPO
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rdkit import DataStructs  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

import pipeline_utils as vs  # noqa: E402
import train_dta_baselines as dta  # noqa: E402
from analyze_external_generalization import fingerprints, tertile_edges  # noqa: E402
from graph_cleaning import clean_graph  # noqa: E402
from evaluate_chembl35 import load_gift_model  # noqa: E402
from evaluate_external import external_matched_negatives  # noqa: E402

OUT = "results/robustness/similarity_strata"


def dta_scores(kind, seed, smiles_ext, ext_off, seqs, pairs):
    ck = torch.load(f"results/gift/dta_leadprio/models/{kind}_seed_{seed}.pth", map_location="cpu",
                    weights_only=False)
    model = dta.build_model(kind, ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    P = torch.tensor(np.stack([dta.label_seq(s, dta.CHARPROTSET, dta.MAX_SEQ) for s in seqs]))
    S = torch.tensor(np.stack([dta.label_seq(s, dta.CHARISOSMISET, dta.MAX_SMI) for s in smiles_ext]))
    graphs = [dta.mol_graph(s) for s in smiles_ext]
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), 2048):
            chunk = pairs[i:i + 2048]
            c = [p[0] - ext_off for p in chunk]
            prot = torch.tensor([p[1] for p in chunk])
            if kind == "deepdta":
                logit = model(S[torch.tensor(c)], P[prot])
            else:
                logit = model(Batch.from_data_list([graphs[j] for j in c]), P[prot])
            out.append(torch.sigmoid(logit).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    seed = ap.parse_args().seed
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    fps = fingerprints()
    d = pickle.load(open(f"data/graphs/graph_seed{seed}.pkl", "rb"))
    gnn_graph = clean_graph(d)[0]
    nc, npr = int(d["num_compounds"]), int(d["num_proteins"])
    compound_ids = list(d["compound_ids"])[:nc]
    protein_ids = list(d["protein_ids"])[:npr]
    train_pairs = vs.derive_chembl35_splits(d, min_training_pchembl=5.0)["train_pairs"]
    gnn_aug, cross_pairs, _ = vs.load_external_chembl36_interactions_from_pkls(gnn_graph)
    aug, cross_d, _ = vs.load_external_chembl36_interactions_from_pkls(d)
    assert cross_d == cross_pairs
    ext_ids = list(aug["external_compound_ids"])
    ext_off = int(aug["external_compound_offset"])
    assert ext_off == nc
    ext_negs = external_matched_negatives(cross_pairs, npr, tested_pairs=aug.get("external_tested_pairs", ()))
    pairs = cross_pairs + ext_negs
    y = np.array([1] * len(cross_pairs) + [0] * len(ext_negs))

    train_cids = {compound_ids[c] for c, _ in train_pairs}
    train_fps = [fps[c] for c in train_cids if c in fps]
    maxsim = {c: max(DataStructs.BulkTanimotoSimilarity(fps[ext_ids[c - ext_off]], train_fps))
              for c in {c for c, _ in pairs}}
    edges = tertile_edges([maxsim[c] for c, _ in pairs])
    strat = np.array([0 if maxsim[c] <= edges[0] else (2 if maxsim[c] > edges[1] else 1) for c, _ in pairs])

    scores = {}
    for name, ck in (("GIFT_main", f"results/gift/run_{seed}/e2e/trained_model.pth"),):
        model, _ = load_gift_model(d, torch.device("cpu"), ck)
        scores[name] = np.asarray(vs.score_gnn_pairs(model, gnn_aug, pairs, torch.device("cpu")), float)

    z = np.load(f"results/gift/baseline_models/drugban/run_{seed}/drugban_dgl/scores.npz")
    assert np.array_equal(z["external|pairs"], np.asarray(pairs)) and np.array_equal(z["external|y"], y)
    scores["DrugBAN"] = z["external|score"].astype(float)
    z = np.load(f"results/robustness/dtiam/runs/run_{seed}/scores.npz")
    assert np.array_equal(z["external|pairs"], np.asarray(pairs)) and np.array_equal(z["external|y"], y)
    scores["DTIAM"] = z["external|score"].astype(float)

    ext_records = pickle.load(open(aug["external_interactions_pkl"], "rb"))["external_compounds"]
    smiles_ext = [ext_records[c]["smiles"] for c in ext_ids]
    seqs_all = json.load(open("data/protein_sequences.json"))
    seqs = [seqs_all[p] for p in protein_ids]
    for kind, name in (("deepdta", "DeepDTA"), ("graphdta", "GraphDTA")):
        scores[name] = dta_scores(kind, seed, smiles_ext, ext_off, seqs, pairs).astype(float)

    rf = joblib.load(f"results/gift/baseline_models/rf/rf_seed_{seed}.joblib")
    rf.n_jobs = 4
    feat = vs.build_pair_feature_fn(aug)
    scores["RF"] = rf.predict_proba(np.stack([feat(c, p) for c, p in pairs]))[:, 1].astype(float)

    res = {"seed": seed, "edges": [float(e) for e in edges], "n_pairs": int(len(y)), "n_pos": int(y.sum()),
           "compound_maxsim": {ext_ids[c - ext_off]: float(v) for c, v in maxsim.items()},
           "overall": {}, "strata": {}}
    for m, s in scores.items():
        res["overall"][m] = {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}
    for b, name in enumerate(("low", "mid", "high")):
        ix = strat == b
        comps = {c for (c, _), k in zip(pairs, ix) if k}
        r = {"n_compounds": len(comps), "n_pairs": int(ix.sum()), "n_pos": int(y[ix].sum()),
             "sim_min": float(min(maxsim[c] for c in comps)), "sim_max": float(max(maxsim[c] for c in comps))}
        for m, s in scores.items():
            r[m] = {"roc_auc": float(roc_auc_score(y[ix], s[ix])), "pr_auc": float(average_precision_score(y[ix], s[ix]))}
        res["strata"][name] = r
    json.dump(res, open(f"{OUT}/seed_{seed}.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/seed_{seed}.npz", pairs=np.asarray(pairs), y=y, strat=strat,
                        **{f"score_{m}": s for m, s in scores.items()})
    print(seed, {m: round(v["roc_auc"], 4) for m, v in res["overall"].items()},
          {b: (r["n_compounds"], r["n_pairs"], round(r["GIFT_main"]["roc_auc"], 3)) for b, r in res["strata"].items()},
          flush=True)


if __name__ == "__main__":
    main()
