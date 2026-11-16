"""Ablations of node masking and contrastive pretraining (supplementary Table S-ablation_phases) and, when finished,
the ESM-2 protein embeddings (supplementary Section S-esm), each compared run by run with the main model (results/gift).

All variants use the main model's settings (Phase-3a learning rate 1e-3). Paired two-sided Wilcoxon signed-rank tests over
the shared seeds 42-71. Inputs: results/gift (main), results/robustness/ablation_phases/<variant>/seed_S/eval_fast.json
(ablation_evaluate.py), results/robustness/esm/run_S/{eval_final,eval_external_matched}.json (scripts/10_train_esm.sh).
Writes results/robustness/ablation_phases/main_comparison.json and, if all ESM runs exist, results/robustness/esm/main_comparison.json.
Usage (repo root): python code/robustness/ablation_main_comparison.py
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

SEEDS = list(range(42, 72))
METRICS = ("roc_auc", "auprc", "mcc")
BENCH = ("same_version", "same_version_matched", "external")


def rd(p):
    return json.JSONDecoder().raw_decode(open(p).read())[0]


def main_metrics(s, b, prefix="results/gift"):
    if b == "external":
        return rd(f"{prefix}/run_{s}/eval_external_matched.json")["results"]["gift"]
    return rd(f"{prefix}/run_{s}/eval_final.json")["results_by_method"]["gift"][b]


def driver_top10(s):
    d = json.load(open(f"results/gift/run_{s}/lead_prioritization.json"))
    return sum(d["per_target"][t]["gnn"]["hits_top"] for t in d["driver_targets"])


def summary(x, ref):
    x, ref = np.asarray(x, float), np.asarray(ref, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "main_mean": float(ref.mean()),
            "main_sd": float(ref.std(ddof=1)), "wilcoxon_p": float(wilcoxon(x, ref).pvalue),
            "runs_higher": int((x > ref).sum()), "runs_lower": int((x < ref).sum()), "n": int(len(x))}


def ablations():
    out = {"reference": "results/gift (main model, Phase-3a learning rate 1e-3)", "variants": {}}
    ref_top10 = [driver_top10(s) for s in SEEDS]
    for v in ("no_mask", "no_pretrain"):
        A = [rd(f"results/robustness/ablation_phases/{v}/seed_{s}/eval_fast.json") for s in SEEDS]
        rec = {b: {m: summary([a["results"][b][m] for a in A], [main_metrics(s, b)[m] for s in SEEDS]) for m in METRICS}
               for b in BENCH}
        rec["driver_top10"] = summary([a["lead"]["driver_top10"] for a in A], ref_top10)
        out["variants"][v] = rec
        print(v)
        for b in BENCH:
            print("  ", b, " | ".join(f"{m} {rec[b][m]['mean']:.3f} (main {rec[b][m]['main_mean']:.3f}, p={rec[b][m]['wilcoxon_p']:.2g})"
                                    for m in METRICS))
        print("   driver top-10", f"{rec['driver_top10']['mean']:.1f} (main {rec['driver_top10']['main_mean']:.1f}, "
                                    f"p={rec['driver_top10']['wilcoxon_p']:.2g})")
    json.dump(out, open("results/robustness/ablation_phases/main_comparison.json", "w"), indent=1)


def esm():
    runs = [s for s in SEEDS if os.path.exists(f"results/robustness/esm/run_{s}/eval_external_matched.json")]
    if len(runs) < len(SEEDS):
        print(f"ESM-2: {len(runs)} of {len(SEEDS)} runs finished; comparison skipped")
        return
    out = {"reference": "results/gift (main model, Phase-3a learning rate 1e-3)"}
    for b in BENCH:
        get = (lambda s: rd(f"results/robustness/esm/run_{s}/eval_external_matched.json")["results"]["gift"]) if b == "external" \
            else (lambda s, b=b: rd(f"results/robustness/esm/run_{s}/eval_final.json")["results_by_method"]["gift"][b])
        out[b] = {m: summary([get(s)[m] for s in SEEDS], [main_metrics(s, b)[m] for s in SEEDS]) for m in METRICS}
        print("ESM-2", b, " | ".join(f"{m} {out[b][m]['mean']:.3f} (main {out[b][m]['main_mean']:.3f}, p={out[b][m]['wilcoxon_p']:.2g})"
                                    for m in METRICS))
    json.dump(out, open("results/robustness/esm/main_comparison.json", "w"), indent=1)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ablations()
    esm()
