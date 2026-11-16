"""Similarity strata (supplementary Table S-similarity): external ROC-AUC / PR-AUC per tertile of the maximum ECFP4
Tanimoto similarity of each external drug to the training compounds of the run, mean +- s.d. over the 30 runs (each
run's value is computed within its own strata; runs are not pooled).
Inputs: results/robustness/similarity_strata/seed_S.json (similarity_strata.py, then similarity_strata_add_dtiam.py).
Writes results/robustness/similarity_strata/summary.json.
Usage (repo root): python code/robustness/similarity_strata_aggregate.py
"""
import json

import numpy as np

D = "results/robustness/similarity_strata"
METHODS = ["GIFT_main", "GraphDTA", "DrugBAN", "DeepDTA", "DTIAM", "RF"]  # GIFT_main = the paper's model
STRATA = ("low", "mid", "high")


def main():
    R = [json.load(open(f"{D}/seed_{s}.json")) for s in range(42, 72)]
    out = {"n_runs": len(R), "strata": {}}
    for b in STRATA:
        rows = [r["strata"][b] for r in R]
        out["strata"][b] = {
            "similarity_range_mean": [float(np.mean([x["sim_min"] for x in rows])), float(np.mean([x["sim_max"] for x in rows]))],
            "n_compounds_mean": float(np.mean([x["n_compounds"] for x in rows])),
            "n_pairs_mean": float(np.mean([x["n_pairs"] for x in rows])),
            "n_pos_mean": float(np.mean([x["n_pos"] for x in rows])),
            "methods": {m: {k: [float(np.mean([x[m][k] for x in rows])), float(np.std([x[m][k] for x in rows], ddof=1))]
                            for k in ("roc_auc", "pr_auc")} for m in METHODS},
        }
    json.dump(out, open(f"{D}/summary.json", "w"), indent=1)
    print("stratum  range         compounds  pairs   positives")
    for b in STRATA:
        o = out["strata"][b]
        print(f"{b:7s}  {o['similarity_range_mean'][0]:.2f}-{o['similarity_range_mean'][1]:.2f}   {o['n_compounds_mean']:6.0f}  "
              f"{o['n_pairs_mean']:7.0f}  {o['n_pos_mean']:7.0f}")
    for m in METHODS:
        print(f"{m:12s} " + " | ".join(
            f"{out['strata'][b]['methods'][m]['roc_auc'][0]:.3f}+-{out['strata'][b]['methods'][m]['roc_auc'][1]:.3f} / "
            f"{out['strata'][b]['methods'][m]['pr_auc'][0]:.3f}+-{out['strata'][b]['methods'][m]['pr_auc'][1]:.3f}"
            for b in STRATA))


if __name__ == "__main__":
    main()
