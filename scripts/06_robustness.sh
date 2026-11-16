#!/usr/bin/env bash
# Review analyses of the paper, recomputed from the saved per-run results (no training; CPU, minutes to ~1 hour).
# Usage: scripts/06_robustness.sh
source "$(dirname "$0")/common.sh"
R=code/robustness
python $R/negative_overlap_audit.py            # held-out positives among training negatives
python $R/measured_negative_counts.py; python $R/measured_activity_corrected.py; python $R/measured_activity_test_only.py
for s in $(seq 42 71); do  # per-run soft-edge score matrices (not stored in the repository; ~15 s each from the saved Phase 2 encoder)
  f=results/gift/run_$s/soft_edge_proposals_scores.npy; t=$(mktemp -d)
  [ -f $f ] || { python code/soft_edge_proposals.py --backbone results/backbones/backbone_seed_$s/best_model_phase2.pth \
    --graph_path "$(graph $s)" --out $t/p.json > /dev/null && mv $t/p_scores.npy $f; }; rm -rf $t
done
python $R/softedge_recovery_test_only.py; python $R/softedge_recovery_per_target.py
python $R/descriptor_and_fixed_budget_main.py; python $R/fixed_budget_main.py   # descriptor-only shortlist, fixed budget
python $R/activity_thresholds.py                                               # pChEMBL >= 5/6/7
python $R/domain_crop_compare_dta.py; python $R/domain_crop_compare_drugban.py
python $R/temporal_and_direct_target.py                                        # temporal and direct-target subsets
python $R/similarity_strata_aggregate.py
python $R/dtiam_numbers.py --json results/robustness/dtiam/numbers.json
python $R/ablation_main_comparison.py                                              # ablations and ESM-2 vs the main model
echo "outputs in results/robustness/<study>/"
