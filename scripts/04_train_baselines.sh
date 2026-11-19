#!/usr/bin/env bash
# Baselines, trained from scratch per run on the same training interactions, negatives rule, validation set,
# and split as GIFT: random forest, GCN/GraphSAGE/GAT, the GCN ladder controls A/B, DeepDTA, GraphDTA, DrugBAN.
# DrugBAN uses the local PyTorch/PyG backend and requirements-drugban.txt in the main environment.
# Usage: DRUGBAN_PY=python [DEVICE=cuda] scripts/04_train_baselines.sh [SEED ...]
source "$(dirname "$0")/common.sh"
DRUGBAN_PY=${DRUGBAN_PY:-${DGL_PY:-}}
B=results/gift/baseline_models; D=results/gift/dta_leadprio
mkdir -p $B/rf $B/gnn $B/controls $B/logs $D/models
for s in "${SEEDS[@]}"; do
  G=$(graph $s)
  python -u code/train_rf_baseline.py --seed $s --out $B/rf > $B/logs/rf_seed_$s.log 2>&1
  python -u code/train_gnn_baselines.py --seed $s --graph_path "$G" \
    --output $B/gnn/gnn_baselines_std_seed_$s.json --save_models $B/gnn > $B/logs/gnn_seed_$s.log 2>&1
  python -u code/train_ladder_controls.py --seed $s --graph_path "$G" --models gcn --decoder dot \
    --output $B/controls/control_A_seed_$s.json --save_models $B/controls > $B/controls/control_A_seed_$s.log 2>&1
  python -u code/train_ladder_controls.py --seed $s --graph_path "$G" --models gcn --decoder fusion \
    --output $B/controls/control_B_seed_$s.json --save_models $B/controls > $B/controls/control_B_seed_$s.log 2>&1
  python -u code/train_dta_baselines.py --seed $s --graph_path "$G" --models deepdta,graphdta \
    --output $D/dta_metrics_seed_$s.json --lead_out $D/dta_lead_seed_$s.json --save_models $D/models > $D/dta_seed_$s.log 2>&1
  if [ -n "$DRUGBAN_PY" ]; then
    d=results/gift/baseline_models/drugban/run_$s; w=$d/drugban_dgl; mkdir -p "$w"
    python code/train_dta_baselines.py --seed $s --graph_path "$G" --output "$w/unused.json" \
      --export_protocol "$w/protocol.pkl" &&
    "$DRUGBAN_PY" code/train_drugban.py --protocol "$w/protocol.pkl" --scores_out "$w/scores.npz"   \
      --device "${DEVICE:-cuda}" &&
    python code/train_dta_baselines.py --seed $s --graph_path "$G" --output "$d/dta_baselines_drugban.json" \
      --import_scores "$w/scores.npz" --scores_out "$d/dta_baselines_drugban_scores.npz" > "$d/dta_baselines_drugban.log" 2>&1 &&
    "$DRUGBAN_PY" code/train_drugban.py --protocol "$w/protocol.pkl" --scores_out "$w/lead_scores.npz"   \
      --checkpoint "$w/scores.npz.ckpt" --lead_out "$w/lead_drugban.json" --device "${DEVICE:-cuda}" > "$w/leadprio.log" 2>&1 &&
    python code/export_drugban_weights.py --out $B/drugban $s
  fi
  echo "baselines seed $s done"
done
