#!/usr/bin/env bash
# Analyses that need the ChEMBL databases or re-scoring with the saved models (run after scripts/02-04 and 07).
# Needs CHEMBL35_DB / CHEMBL36_DB for the temporal records. Usage: 11_extra_analyses.sh [seeds]
source "$(dirname "$0")/common.sh"
DRUGBAN_PY=${DRUGBAN_PY:-${DGL_PY:-}}
R=code/robustness
[ -f results/robustness/temporal_direct_target/records_c36.pkl ] || python $R/temporal_extract_chembl_records.py ${CHEMBL36_DB:+--c36 "$CHEMBL36_DB"} ${CHEMBL35_DB:+--c35 "$CHEMBL35_DB"}
for s in "${SEEDS[@]}"; do
  python $R/temporal_predict_pairs.py --seed $s      # frozen-model scores of the release-36-first pairs
  python $R/similarity_strata.py --seed $s           # per-run scores in the low/medium/high similarity strata
done
python $R/similarity_strata_add_dtiam.py; python $R/similarity_strata_aggregate.py
python $R/fixed_budget_score_baselines.py --seeds "${SEEDS[@]}"; python $R/fixed_budget_score_gift_main.py --kind run --seeds "${SEEDS[@]}"
python $R/fixed_budget_score_gift_main.py --kind descriptor_only --seeds "${SEEDS[@]}"
python $R/fixed_budget_score_dtiam_val.py
[ -n "$DRUGBAN_PY" ] && "$DRUGBAN_PY" $R/fixed_budget_score_drugban_val.py --seeds "${SEEDS[@]}"
python $R/truncation_audit.py   # kinase-domain positions (needs CHEMBL36_DB)
python $R/negative_overlap_verify_replay.py
bash "$(dirname "$0")/06_robustness.sh"
