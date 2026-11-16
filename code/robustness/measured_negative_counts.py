"""Point 2: reproduce and decompose the '564' (Appendix A) and '2,257' (Table S15) below-threshold pair counts.

Read-only; CPU. Uses the per-run graph pickles data/graphs/graph_seed{s}.pkl (built by
build_run_graphs.py, which stores all_known_pchembl = median pChEMBL per recorded pair, or 0.0 when the
pair has no pChEMBL value; line `out["all_known_pchembl"] = {p: median.get(p, 0.0) for p in recorded}`) and the
selection rule of eval_measured_activity.py (negatives: med < 5 / 3 <= med < 5, not training, in the
20% evaluation reserve of negative_pool.reserved_eval_mask).
Usage: python code/robustness/measured_negative_counts.py   (from the repo root)
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from negative_pool import reserved_eval_mask  # noqa: E402

OUT = "results/robustness/measured_activity_thresholds/pt2_counts.json"


def main():
    per_seed = {}
    for s in range(42, 72):
        g = pickle.load(open(f"data/graphs/graph_seed{s}.pkl", "rb"))
        med = {tuple(map(int, k)): float(v) for k, v in dict(g["all_known_pchembl"]).items()}
        has_pc = {tuple(map(int, k)) for k in g["pchembl_n_records"]}
        nc, npr = int(g["num_compounds"]), int(g["num_proteins"])
        R = reserved_eval_mask(nc, npr)
        so = g["split_override"]
        tr = {tuple(map(int, p)) for p in so["train"]}
        val = {tuple(map(int, p)) for p in so["val"]}
        te = {tuple(map(int, p)) for p in so["test"]}
        lt5 = [p for p in med if med[p] < 5]
        lt5_pc = [p for p in lt5 if p in has_pc]
        neg_all = [p for p in lt5 if p not in tr and R[p]]
        neg_mod = [p for p in med if 3 <= med[p] < 5 and p not in tr and R[p]]
        per_seed[s] = {
            "recorded_pairs": len(med), "with_pchembl": len(has_pc), "without_pchembl": len(med) - len(has_pc),
            "median_ge5": sum(v >= 5 for v in med.values()),
            "median_lt5_incl_no_pchembl": len(lt5),
            "with_pchembl_median_lt5": len(lt5_pc),
            "with_pchembl_median_in_3_5": sum(1 for p in lt5_pc if med[p] >= 3),
            "with_pchembl_median_lt3": sum(1 for p in lt5_pc if med[p] < 3),
            "reserve_fraction": float(R.mean()),
            "table_s15_all_negatives": len(neg_all),
            "table_s15_all_with_pchembl": sum(p in has_pc for p in neg_all),
            "table_s15_all_without_pchembl": sum(p not in has_pc for p in neg_all),
            "table_s15_moderate_negatives": len(neg_mod),
            "with_pchembl_lt5_outside_reserve": sum(1 for p in lt5_pc if not R[p]),
            "without_pchembl_in_reserve": sum(1 for p in lt5 if p not in has_pc and R[p]),
            "n_train": len(tr), "n_val": len(val), "n_test": len(te),
            "lt5_pairs_in_any_split": sum(p in tr | val | te for p in lt5),
        }
    keys = per_seed[42].keys()
    const = {k: len({per_seed[s][k] for s in per_seed}) == 1 for k in keys}
    summary = {k: (per_seed[42][k] if const[k] else [min(per_seed[s][k] for s in per_seed),
                                                         max(per_seed[s][k] for s in per_seed)]) for k in keys}
    json.dump({"summary_seed_invariant_or_range": summary, "constant_across_seeds": const, "per_seed": per_seed},
              open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
