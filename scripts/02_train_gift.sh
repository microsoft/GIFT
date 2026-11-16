#!/usr/bin/env bash
# GIFT, one run per seed: Phase 1-2 backbone (contrastive pretraining + soft edges), Phase 3a classifier,
# Phase 3b end-to-end fine-tuning, evaluation on the three benchmarks, lead prioritization, misclassification
# analysis (t-SNE), and soft-edge proposal statistics.
# Usage: scripts/02_train_gift.sh [SEED ...]
source "$(dirname "$0")/common.sh"
for s in "${SEEDS[@]}"; do
  G=$(graph $s); bb=results/backbones/backbone_seed_$s; d=results/gift/run_$s
  mkdir -p "$bb" "$d/misclassification"
  python -u code/train_backbone.py --seed $s --results_dir "$bb" --graph_path "$G" \
    --num_epochs_phase1 200 --num_epochs_phase2 200 --phase2_attention_batch_size 100 > "$bb/train.log" 2>&1 &&
  python -u code/train_classifier.py --seed $s --results_dir "$d/head" --graph_path "$G" --neg_ratio 1 \
    --phase2_checkpoint "$bb/best_model_phase2.pth" $PM > "$d/head.log" 2>&1 &&
  python -u code/train_end_to_end.py --seed $s --results_dir "$d/e2e" --graph_path "$G" --neg_ratio 1 \
    --init_checkpoint "$d/head/trained_model.pth" $PM > "$d/e2e.log" 2>&1 &&
  python code/evaluate_chembl35.py --seed $s --graph_path "$G" \
    --model_checkpoint "$d/e2e/trained_model.pth" --output "$d/eval_final.json" > "$d/eval.log" 2>&1 &&
  python code/evaluate_external.py --seed $s --graph_path "$G" \
    --checkpoint "$d/e2e/trained_model.pth" --output "$d/eval_external_matched.json" > "$d/eval_external_matched.log" 2>&1 &&
  python code/lead_prioritization.py --seed $s --run_dir "$d" --graph_path "$G" > "$d/lead_prioritization.log" 2>&1 &&
  python code/analyze_misclassification.py --run_dir "$d" --graph_path "$G" \
    --out "$d/misclassification/tsne_pairs_predictions" > "$d/misclassification_random.log" 2>&1 &&
  python code/soft_edge_proposals.py --backbone "$bb/best_model_phase2.pth" --graph_path "$G" \
    --out "$d/soft_edge_proposals.json" > "$d/soft_edge_proposals.log" 2>&1
  echo "GIFT seed $s exit=$?"
done
