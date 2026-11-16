"""Fixed candidate budget and validation-chosen threshold (supplementary Table S-fixed_budget) for the paper's model.

Runs point 5 of descriptor_and_fixed_budget.py on the per-pair scores of the paper's model and every baseline
(results/robustness/descriptor_fixed_budget/scores_main_combined: GIFT, descriptor-only, RF, GCN, GraphSAGE, GAT,
DeepDTA, GraphDTA; DrugBAN and DTIAM benchmark scores are read from their runs, their validation scores from
fixed_budget_score_drugban_val.py and fixed_budget_score_dtiam_val.py). Writes results_pt5_main.json next to the scores.
Usage (repo root): python code/robustness/fixed_budget_main.py
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import json  # noqa: E402

import numpy as np  # noqa: E402

import descriptor_and_fixed_budget as A  # noqa: E402

D = "results/robustness/descriptor_fixed_budget"


def main():
    A.SC = _os.path.join(D, "scores_main_combined")
    p5 = A.point5()
    json.dump(p5, open(f"{D}/results_pt5_main.json", "w"), indent=1,
              default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    for b in A.BENCHES:
        print(f"== {b}")
        for m in A.MODELS:
            r, t = p5["aggregate"][b][m], p5["wilcoxon_gift_vs"][b].get(m, {})
            hk = "  ".join(f"@{k} {r[f'hits@{k}']['mean']:.1f}" for k in A.KS)
            print(f"  {m:16s} {hk} | val-MCC threshold: MCC {r['valmcc_mcc']['mean']:.3f}"
                  + (f" (GIFT vs: p@100={t['hits@100']:.0e})" if t else ""))


if __name__ == "__main__":
    main()
