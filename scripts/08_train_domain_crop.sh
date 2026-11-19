#!/usr/bin/env bash
# Sequence-truncation study: DeepDTA, GraphDTA and DrugBAN with the original N-terminal prefix (nocrop) and with
# domain-preserving crops (crop). Needs CHEMBL36_DB for crops, DRUGBAN_PY=python for DrugBAN. Usage: 08_train_domain_crop.sh [seeds]
source "$(dirname "$0")/common.sh"
DRUGBAN_PY=${DRUGBAN_PY:-${DGL_PY:-}}
R=results/robustness/domain_crop
python code/robustness/truncation_audit.py
[ -f $R/domain_crops.json ] || python code/robustness/domain_crop_build.py
for MODE in crop nocrop; do
  CROP=""; [ $MODE = crop ] && CROP="--domain_crop $R/domain_crops.json"
  mkdir -p $R/dta_$MODE/models
  for s in "${SEEDS[@]}"; do
    G=$(graph $s)
    python -u code/robustness/domain_crop_train_dta.py --seed $s --graph_path "$G" --models deepdta,graphdta $CROP \
      --output $R/dta_$MODE/dta_metrics_seed_$s.json --lead_out $R/dta_$MODE/dta_lead_seed_$s.json \
      --save_models $R/dta_$MODE/models > $R/dta_$MODE/dta_seed_$s.log 2>&1
    if [ -n "$DRUGBAN_PY" ]; then
      d=$R/drugban_$MODE/run_$s; w=$d/drugban_dgl; mkdir -p "$w"
      python code/robustness/domain_crop_train_dta.py --seed $s --graph_path "$G" $CROP --output "$w/unused.json" --export_protocol "$w/protocol.pkl" &&
      "$DRUGBAN_PY" code/robustness/domain_crop_train_drugban.py --protocol "$w/protocol.pkl" --scores_out "$w/scores.npz" --device "${DEVICE:-cuda}" &&
      python code/robustness/domain_crop_train_dta.py --seed $s --graph_path "$G" $CROP --output "$d/dta_baselines_drugban.json" \
        --import_scores "$w/scores.npz" --scores_out "$d/dta_baselines_drugban_scores.npz" > "$d/dta_baselines_drugban.log" 2>&1 &&
      "$DRUGBAN_PY" code/robustness/domain_crop_train_drugban.py --protocol "$w/protocol.pkl" --scores_out "$w/lead_scores_unused.npz"   \
        --checkpoint "$w/scores.npz.ckpt" --lead_out "$w/lead_drugban.json" --device "${DEVICE:-cuda}"
    fi
    echo "domain crop $MODE seed $s done"
  done
done
python code/robustness/domain_crop_compare_dta.py; [ -n "$DRUGBAN_PY" ] && python code/robustness/domain_crop_compare_drugban.py
