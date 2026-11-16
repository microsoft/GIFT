"""Target-specificity analysis: does conditioning on the target beat a target-blind ranking?

For each driver kinase, rank the 95 external drugs by three rankings and record the #1-ranked drug:
  - gnn          : the model's target-conditioned ranking (a different ranking per target);
  - gnn_family   : the kinase-family comparator (one ranking shared by all kinases);
  - gnn_shared   : the global comparator (one ranking shared by every target).
Reports, per seed and aggregated over seeds 42-71:
  - drivers_active_at_1 : how many driver kinases have a known active drug ranked #1;
  - distinct_drugs_at_1 : how many distinct drugs occupy those #1 positions.
A target-blind ranking puts the same drug first for every driver (distinct == 1 by construction),
so this isolates whether the model's target conditioning improves the #1 decision.

Reuses the exact loaders of lead_prioritization.py for consistency. Inference only.
"""

import argparse
import json
import pickle

import numpy as np
import torch

import pipeline_utils as vs
from graph_cleaning import clean_graph
from evaluate_chembl35 import load_gift_model

MIN_ACTIVES = 5
DRIVERS = {
    "CHEMBL203": "EGFR", "CHEMBL1824": "ERBB2", "CHEMBL4247": "ALK", "CHEMBL3717": "MET",
    "CHEMBL2041": "RET", "CHEMBL5568": "ROS1", "CHEMBL5145": "BRAF",
    "CHEMBL2815": "NTRK1", "CHEMBL4898": "NTRK2", "CHEMBL5608": "NTRK3",
}
KINASE_PATH = "data/protein_kinase_annotation.json"


def one_seed(seed, graph_path, run_dir, device):
    with open(graph_path, "rb") as f:
        graph_dict = pickle.load(f)
    model, _ = load_gift_model(graph_dict, device, f"{run_dir}/e2e/trained_model.pth")
    gnn_graph = clean_graph(graph_dict)[0]

    num_proteins = int(graph_dict["num_proteins"])
    protein_ids = list(graph_dict["protein_ids"])[:num_proteins]

    gnn_aug, gnn_cross_pairs, _ = vs.load_external_chembl36_interactions_from_pkls(gnn_graph)
    ext_idx = sorted({c for c, _ in gnn_cross_pairs})
    n_ext = len(ext_idx)
    col = {c: i for i, c in enumerate(ext_idx)}

    labels = np.zeros((num_proteins, n_ext), dtype=np.int64)
    for c, p in gnn_cross_pairs:
        labels[p, col[c]] = 1

    all_pairs = [(c, p) for p in range(num_proteins) for c in ext_idx]
    gnn = np.asarray(vs.score_gnn_pairs(model, gnn_aug, all_pairs, device)).reshape(num_proteins, n_ext)

    kinase_kw = json.load(open(KINASE_PATH))
    is_kinase = np.array([kinase_kw.get(t, {}).get("kinase", False) for t in protein_ids])

    n_act = labels.sum(axis=1)
    drivers = [p for p in range(num_proteins)
               if protein_ids[p] in DRIVERS and MIN_ACTIVES <= n_act[p] < n_ext]

    # One shared ranking vector per comparator (same 95-drug ordering for every target).
    global_vec = gnn.mean(axis=0)                 # mean over all proteins
    kinase_vec = gnn[is_kinase].mean(axis=0)      # mean over kinase-family proteins

    out = {}
    for name, scorer in (("gnn", lambda p: gnn[p]),
                         ("gnn_family", lambda p: kinase_vec),
                         ("gnn_shared", lambda p: global_vec)):
        top_drug = {}
        active_at_1 = 0
        for p in drivers:
            i1 = int(np.argmax(scorer(p)))       # stable: argmax picks first max
            top_drug[p] = ext_idx[i1]
            active_at_1 += int(labels[p, i1] == 1)
        out[name] = {
            "drivers_active_at_1": active_at_1,
            "distinct_drugs_at_1": len(set(top_drug.values())),
            "n_drivers": len(drivers),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 72)))
    ap.add_argument("--graph_tmpl", default="data/graphs/graph_seed{s}.pkl")
    ap.add_argument("--run_tmpl", default="results/gift/run_{s}")
    ap.add_argument("--output", default="results/gift/target_specificity.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed = {}
    for s in args.seeds:
        r = one_seed(s, args.graph_tmpl.format(s=s), args.run_tmpl.format(s=s), device)
        per_seed[s] = r
        print(f"seed {s}: " + " | ".join(
            f"{m}: active@1={r[m]['drivers_active_at_1']}/{r[m]['n_drivers']} "
            f"distinct@1={r[m]['distinct_drugs_at_1']}" for m in ("gnn", "gnn_family", "gnn_shared")))

    agg = {}
    for m in ("gnn", "gnn_family", "gnn_shared"):
        a = np.array([per_seed[s][m]["drivers_active_at_1"] for s in args.seeds], float)
        d = np.array([per_seed[s][m]["distinct_drugs_at_1"] for s in args.seeds], float)
        agg[m] = {
            "drivers_active_at_1_mean": float(a.mean()), "drivers_active_at_1_std": float(a.std()),
            "distinct_drugs_at_1_mean": float(d.mean()), "distinct_drugs_at_1_std": float(d.std()),
        }
    json.dump({"per_seed": per_seed, "agg": agg}, open(args.output, "w"), indent=2)
    print("\n=== aggregate over", len(args.seeds), "seeds ===")
    for m in ("gnn", "gnn_family", "gnn_shared"):
        g = agg[m]
        print(f"{m:12s} active@1 {g['drivers_active_at_1_mean']:.2f}±{g['drivers_active_at_1_std']:.2f}"
              f"   distinct@1 {g['distinct_drugs_at_1_mean']:.2f}±{g['distinct_drugs_at_1_std']:.2f}")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
