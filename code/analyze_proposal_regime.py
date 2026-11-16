"""Exploratory soft-edge proposal-regime analysis: does the proposal regime explain the benefit?

Question: do runs that propose interaction-like edges improve predictions more than runs whose
proposals do not? For each of the 30 main runs we read, from files that already exist:
  - proposal interaction-likeness: ROC-AUC of the soft-edge compound-protein proposal scores for recovering
    the held-out interactions among all candidate pairs (soft_edge_proposals.json -> roc_auc_all_candidates),
    and the held-out enrichment of the top-k proposals;
  - the per-seed soft-edge BENEFIT = main minus the no-soft-edge ablation on three endpoints:
      driver top-1 recovery (active drug ranked #1, computed from the stored driver_scores/driver_labels),
      ChEMBL35 compound-matched MCC, and external ROC-AUC.
Then it reports Spearman correlations between proposal interaction-likeness and each benefit across seeds.
No training and no model inference (everything is read from the saved run outputs).
"""

import glob
import json
import os
import re

import numpy as np
from scipy.stats import spearmanr

R = "results/gift"


def active_at_1(lead_json):
    d = json.load(open(lead_json))
    ds, dl = d["driver_scores"], d["driver_labels"]
    n = 0
    for t, scores in ds.items():
        top = max(scores, key=scores.get)
        n += int(top in set(dl[t]))
    return n


def matched_mcc(eval_final):
    r = json.load(open(eval_final))["results_by_method"]["gift"]
    return r["same_version_matched"]["mcc"]


def external_roc(eval_ext):
    return json.load(open(eval_ext))["results"]["gift"]["roc_auc"]


def main():
    seeds = sorted(int(re.search(r"run_(\d+)$", p).group(1))
                   for p in glob.glob(f"{R}/run_*") if re.search(r"run_(\d+)$", p))
    rows = []
    for s in seeds:
        prop = json.load(open(f"{R}/run_{s}/soft_edge_proposals.json"))
        g = prop["global"]
        main_dir, abl_dir = f"{R}/run_{s}", f"{R}/ablation_no_softedge_seed_{s}"
        if not os.path.exists(f"{abl_dir}/eval_final.json"):
            continue
        rows.append({
            "seed": s,
            "prop_roc": g.get("roc_auc_all_candidates", float("nan")),
            "prop_enrich": g.get("enrichment", float("nan")),
            "prop_pt_roc": prop["per_target_summary"].get("mean_roc_auc", float("nan")),
            "d_top1": active_at_1(f"{main_dir}/lead_prioritization.json") - active_at_1(f"{abl_dir}/lead_prioritization.json"),
            "d_matched_mcc": matched_mcc(f"{main_dir}/eval_final.json") - matched_mcc(f"{abl_dir}/eval_final.json"),
            "d_ext_roc": external_roc(f"{main_dir}/eval_external_matched.json") - external_roc(f"{abl_dir}/eval_external_matched.json"),
        })

    arr = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    print(f"seeds: {len(rows)}")
    print("\nmean soft-edge benefit (main - no-soft-edge):")
    for k, lab in (("d_top1", "driver top-1 recovery"), ("d_matched_mcc", "compound-matched MCC"), ("d_ext_roc", "external ROC-AUC")):
        print(f"  {lab:24s} {arr[k].mean():+.3f} ± {arr[k].std():.3f}")
    print("\nproposal interaction-likeness across seeds:")
    print(f"  roc_auc_all_candidates  {arr['prop_roc'].mean():.3f} ± {arr['prop_roc'].std():.3f} "
          f"(min {arr['prop_roc'].min():.3f}, max {arr['prop_roc'].max():.3f})")
    print(f"  top-k enrichment        {arr['prop_enrich'].mean():.2f} ± {arr['prop_enrich'].std():.2f} "
          f"(max {arr['prop_enrich'].max():.2f})")

    print("\nSpearman correlation (proposal interaction-likeness vs benefit):")
    for pk, pl in (("prop_roc", "roc_auc_all_candidates"), ("prop_enrich", "top-k enrichment"), ("prop_pt_roc", "per-target proposal ROC")):
        for bk, bl in (("d_top1", "driver top-1"), ("d_matched_mcc", "matched MCC"), ("d_ext_roc", "external ROC")):
            x, y = arr[pk], arr[bk]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 5 and np.ptp(x[m]) > 0 and np.ptp(y[m]) > 0:
                rho, p = spearmanr(x[m], y[m])
                print(f"  {pl:24s} vs {bl:14s}: rho={rho:+.3f}  p={p:.3f}  (n={m.sum()})")

    json.dump({"rows": rows}, open(f"{R}/proposal_regime.json", "w"), indent=2)
    print(f"\nwrote {R}/proposal_regime.json")


if __name__ == "__main__":
    main()
