"""Attribution and lead-prioritization numbers for the manuscript, paired by seed (30 runs).

- Lead prioritization by method: driver kinases with a known active ranked first and known actives in the top 10
  (summed over drivers) for the proposed model, DeepDTA/GraphDTA (results/gift/dta_leadprio) and DrugBAN.
- Component ladder: A = standard GCN + dot decoder, B = GCN + fusion classifier (both matched in label exposure),
  D = proposed model; MCC on the compound-matched and external benchmarks and external ROC-AUC.
- Descriptor-only, random-augmentation and no-soft-edge controls vs the proposed model on every benchmark.
- Seed-42 soft-edge comparison for the top-picks figure (known actives in the top 10 and per-target ROC-AUC).
Usage: python attribution_numbers.py --prop results/gift
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

SEEDS = range(42, 72)
M = "gift"
BENCH = ["same_version", "same_version_isolated", "same_version_matched", "external"]
METRICS = ["roc_auc", "auprc", "per_target_roc_auc", "precision", "recall", "f1", "mcc"]


def ms(v, d=3):
    return f"{np.mean(v):.{d}f}$\\pm${np.std(v, ddof=1):.{d}f}"


def paired(a, b, lower=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    better = int(((a < b) if lower else (a > b)).sum())
    worse = int(((a > b) if lower else (a < b)).sum())
    p = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
    return f"{better}/{worse}/{len(a) - better - worse} (+/-/=) p={p:.2g}"


def lead_counts(path):
    d = json.load(open(path))
    t1 = t10 = 0
    for k in d["driver_scores"]:
        acts = set(d["driver_labels"][k])
        order = sorted(d["driver_scores"][k], key=lambda c: -d["driver_scores"][k][c])
        t1 += order[0] in acts
        t10 += sum(c in acts for c in order[:10])
    return t1, t10


def bench(run_dir, b):
    if b == "external":
        return json.load(open(f"{run_dir}/eval_external_matched.json"))["results"][M]
    return json.load(open(f"{run_dir}/eval_final.json"))["results_by_method"][M][b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", required=True)
    R = ap.parse_args().prop

    print("== Lead prioritization by method (driver kinases)")
    prop = np.array([lead_counts(f"{R}/run_{s}/lead_prioritization.json") for s in SEEDS], float)
    base = {}
    for m in ("deepdta", "graphdta"):
        base[m] = np.array([[json.load(open(f"results/gift/dta_leadprio/dta_lead_seed_{s}.json"))["lead"][m][k]
                             for k in ("drivers_active_at_1", "hits_top10_sum")] for s in SEEDS], float)
    base["drugban"] = np.array([[json.load(open(f"results/gift/baseline_models/drugban/run_{s}/drugban_dgl/lead_drugban.json"))["lead"]["drugban"][k]
                                 for k in ("drivers_active_at_1", "hits_top10_sum")] for s in SEEDS], float)
    print(f"Proposed   top1 {ms(prop[:, 0], 1)}  top10 {ms(prop[:, 1], 1)}")
    for m, v in base.items():
        print(f"{m:10s} top1 {ms(v[:, 0], 1)}  top10 {ms(v[:, 1], 1)}   proposed vs {m}: "
              f"top1 {paired(prop[:, 0], v[:, 0])}; top10 {paired(prop[:, 1], v[:, 1])}")

    print("\n== Component ladder (A: GCN+dot, B: GCN+fusion, D: proposed)")
    def gcn(f, b):
        return json.load(open(f))["results"]["gcn"]["cross_version" if b == "external" else b]
    for b, k in [("same_version_matched", "mcc"), ("external", "mcc"), ("external", "roc_auc"),
                 ("same_version", "mcc"), ("same_version_matched", "roc_auc")]:
        # Saved-weight controls (scripts/04_train_baselines.sh); the original unsaved runs are control_A_gcn_dot_seed_*.json
        # and matched_gcn_seed_*.json in results/gift.
        A = [gcn(f"results/gift/baseline_models/controls/control_A_seed_{s}.json", b)[k] for s in SEEDS]
        B = [gcn(f"results/gift/baseline_models/controls/control_B_seed_{s}.json", b)[k] for s in SEEDS]
        D = [bench(f"{R}/run_{s}", b)[k] for s in SEEDS]
        print(f"{b:22s} {k:8s} A {np.mean(A):.3f}  B {np.mean(B):.3f}  D {np.mean(D):.3f}   "
              f"A->B {paired(B, A)}   B->D {paired(D, B)}")

    for ctrl in ("descriptor_only", "random_aug", "ablation_no_softedge"):
        print(f"\n== {ctrl} vs proposed (proposed better/worse/equal)")
        for b in BENCH:
            for k in METRICS:
                try:
                    c = [bench(f"{R}/{ctrl}_seed_{s}", b)[k] for s in SEEDS]
                except (KeyError, FileNotFoundError):
                    continue
                p = [bench(f"{R}/run_{s}", b)[k] for s in SEEDS]
                print(f"{b:22s} {k:20s} proposed {ms(p)}  {ctrl} {ms(c)}  {paired(p, c)}")
        if all(os.path.exists(f"{R}/{ctrl}_seed_{s}/lead_prioritization.json") for s in SEEDS):
            c = np.array([lead_counts(f"{R}/{ctrl}_seed_{s}/lead_prioritization.json") for s in SEEDS], float)
            print(f"driver top1 proposed {ms(prop[:, 0], 2)} {ctrl} {ms(c[:, 0], 2)} {paired(prop[:, 0], c[:, 0])}")
            print(f"driver top10 proposed {ms(prop[:, 1], 1)} {ctrl} {ms(c[:, 1], 1)} {paired(prop[:, 1], c[:, 1])}")

    print("\n== Seed 42: with vs without soft edges (driver kinases)")
    w = json.load(open(f"{R}/run_42/lead_prioritization.json"))
    wo = json.load(open(f"{R}/ablation_no_softedge_seed_42/lead_prioritization.json"))
    more = same = fewer = auc_hi = 0
    tw = two = 0
    for t in w["driver_scores"]:
        a, b = w["per_target"][t]["gnn"], wo["per_target"][t]["gnn"]
        tw += a["hits_top"]; two += b["hits_top"]
        more += a["hits_top"] > b["hits_top"]; same += a["hits_top"] == b["hits_top"]
        fewer += a["hits_top"] < b["hits_top"]; auc_hi += a["roc_auc"] > b["roc_auc"]
        print(f"  {t:12s} top10 {a['hits_top']} vs {b['hits_top']}   ROC-AUC {a['roc_auc']:.3f} vs {b['roc_auc']:.3f}")
    print(f"  more {more}, same {same}, fewer {fewer}; total {tw} vs {two}; ROC-AUC higher for {auc_hi} of 10")


if __name__ == "__main__":
    main()
