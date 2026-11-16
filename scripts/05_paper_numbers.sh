#!/usr/bin/env bash
# Every table, figure, and number of the paper from the per-run results (no training; minutes on CPU).
# Usage: scripts/05_paper_numbers.sh
source "$(dirname "$0")/common.sh"
R=results/gift; mkdir -p figures
python code/audit_splits.py                                    | tail -1   # split and leakage audit
python code/paper_numbers.py --prop $R --json $R/paper_numbers_main_table.json > $R/paper_numbers.txt  # Tab. main, DTA, confusion
python code/attribution_numbers.py --prop $R > $R/attribution_numbers.txt 2>&1  # ladder, controls, lead by method
python code/analyze_target_specificity.py > $R/target_specificity.log 2>&1      # family vs per-target rankings
python code/analyze_external_generalization.py > $R/external_generalization.log 2>&1
python code/eval_measured_activity.py > $R/measured_activity.log 2>&1
python code/analyze_proposal_regime.py > $R/proposal_regime.log 2>&1
python code/make_lead_figures.py --prop $R --outdir figures                 # Figure 2 and Supplementary Figure S4
python code/make_value_figures.py --outdir figures                           # Supplementary Figures S2 and S5
python code/make_tsne_figure.py --csv $R/run_42/misclassification/pairs_random.csv \
  --out figures/tsne_pairs_predictions --left_label "" --box_label ""                 # supplementary t-SNE
echo "outputs in $R/*.txt|*.json and figures/"
