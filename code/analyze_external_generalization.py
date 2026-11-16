"""External generalization stratification: where does external performance hold?

For the full model on the external ChEMBL36 benchmark (the 3,243 held-out actives of the 95 drugs against
compound-matched negatives), this stratifies the pairs three ways and reports per-stratum ranking quality:
  (a) compound similarity to training: max ECFP4 Tanimoto of the drug to any training compound (per seed);
  (b) target coverage: number of training interactions of the target;
  (c) target class: kinase vs non-kinase.
Metric per stratum: pooled ROC-AUC over its pairs, averaged over seeds 42-71. Inference only.
"""

import json
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score

import pipeline_utils as vs
from graph_cleaning import clean_graph
from evaluate_chembl35 import load_gift_model
from evaluate_external import external_matched_negatives

KINASE_PATH = "data/protein_kinase_annotation.json"
SEEDS = list(range(42, 72))


def fingerprints():
    """ECFP4 (radius 2, 2048 bit) for every graph and external compound, keyed by compound id."""
    d = pickle.load(open("data/graphs/graph_seed42.pkl", "rb"))
    aug, _, _ = vs.load_external_chembl36_interactions_from_pkls(d)
    smi = pd.read_csv("data/chembl_lung_cancer_all_compounds.csv").dropna(subset=["smiles"])
    smi = smi.drop_duplicates("compound_id").set_index("compound_id")["smiles"].to_dict()
    ext_records = pickle.load(open(aug["external_interactions_pkl"], "rb"))["external_compounds"]
    for c, r in ext_records.items():
        smi[c] = r["smiles"]
    fps = {}
    for c, s in smi.items():
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            fps[c] = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    return fps


def tertile_edges(vals):
    v = np.asarray(vals, float)
    return np.quantile(v, [1 / 3, 2 / 3])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = fingerprints()
    kinase_kw = json.load(open(KINASE_PATH))

    # Accumulate per-stratum (scores, labels) pooled across seeds.
    buckets = {("sim", b): ([], []) for b in ("low", "mid", "high")}
    buckets.update({("cov", b): ([], []) for b in ("low", "mid", "high")})
    buckets.update({("kin", b): ([], []) for b in ("kinase", "non-kinase")})

    for seed in SEEDS:
        d = pickle.load(open(f"data/graphs/graph_seed{seed}.pkl", "rb"))
        model, _ = load_gift_model(d, device, f"results/gift/run_{seed}/e2e/trained_model.pth")
        gnn_graph = clean_graph(d)[0]
        nc, npr = int(d["num_compounds"]), int(d["num_proteins"])
        compound_ids = list(d["compound_ids"])[:nc]
        protein_ids = list(d["protein_ids"])[:npr]
        splits = vs.derive_chembl35_splits(d, min_training_pchembl=5.0)
        train_pairs = splits["train_pairs"]

        gnn_aug, cross_pairs, _ = vs.load_external_chembl36_interactions_from_pkls(gnn_graph)
        aug, _, _ = vs.load_external_chembl36_interactions_from_pkls(d)
        ext_ids = list(aug["external_compound_ids"])
        ext_off = int(aug["external_compound_offset"])
        ext_negs = external_matched_negatives(cross_pairs, npr, tested_pairs=aug.get("external_tested_pairs", ()))
        pairs = cross_pairs + ext_negs
        y = np.array([1] * len(cross_pairs) + [0] * len(ext_negs))
        s = vs.score_gnn_pairs(model, gnn_aug, pairs, device)

        # (a) per external compound: max Tanimoto to any TRAINING compound of this seed.
        train_cids = {compound_ids[c] for c, _ in train_pairs if c < nc}
        train_fps = [fps[c] for c in train_cids if c in fps]
        def resolve(c):
            return compound_ids[c] if c < nc else ext_ids[c - ext_off]
        maxsim = {}
        for c in {c for c, _ in pairs}:
            fp = fps.get(resolve(c))
            maxsim[c] = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) if fp is not None and train_fps else 0.0
        # (b) target coverage (train interactions per protein).
        cov = np.bincount([p for _, p in train_pairs], minlength=npr)
        # (c) kinase flag.
        is_kin = np.array([kinase_kw.get(t, {}).get("kinase", False) for t in protein_ids])

        sim_e = tertile_edges([maxsim[c] for c, _ in pairs])
        cov_e = tertile_edges([cov[p] for _, p in pairs])
        for i, (c, p) in enumerate(pairs):
            sb = "low" if maxsim[c] <= sim_e[0] else ("high" if maxsim[c] > sim_e[1] else "mid")
            cb = "low" if cov[p] <= cov_e[0] else ("high" if cov[p] > cov_e[1] else "mid")
            kb = "kinase" if is_kin[p] else "non-kinase"
            for key in (("sim", sb), ("cov", cb), ("kin", kb)):
                buckets[key][0].append(s[i]); buckets[key][1].append(y[i])
        print(f"seed {seed} done", flush=True)

    print("\n=== External generalization by stratum (pooled ROC-AUC over 30 seeds) ===")
    labels = {"sim": "compound max-Tanimoto to train", "cov": "target training coverage", "kin": "target class"}
    out = {}
    for fac in ("sim", "cov", "kin"):
        print(f"\n[{labels[fac]}]")
        for b in (("low", "mid", "high") if fac != "kin" else ("kinase", "non-kinase")):
            sc, yy = buckets[(fac, b)]
            yy = np.array(yy)
            auc = roc_auc_score(yy, np.array(sc)) if 0 < yy.sum() < len(yy) else float("nan")
            out[f"{fac}:{b}"] = {"roc_auc": float(auc), "n_pos": int(yy.sum()), "n": int(len(yy))}
            print(f"  {b:11s} ROC-AUC={auc:.3f}  (pairs={len(yy)}, pos={int(yy.sum())})")
    json.dump(out, open("results/gift/external_generalization.json", "w"), indent=2)
    print("\nwrote results/gift/external_generalization.json")


if __name__ == "__main__":
    main()
