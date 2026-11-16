#!/usr/bin/env bash
# Attribution controls (needs the backbones of 02_train_gift.sh): GIFT without soft edges, descriptor-only fusion,
# and matched random graph augmentation. Usage: scripts/03_train_controls.sh [SEED ...]
source "$(dirname "$0")/common.sh"
R=results/gift
evaluate() {  # $1 run dir, $2 graph, $3 seed, $4 = lead to also run lead prioritization
  python code/evaluate_chembl35.py --seed $3 --graph_path "$2" \
    --model_checkpoint "$1/e2e/trained_model.pth" --output "$1/eval_final.json" > "$1/eval.log" 2>&1 &&
  python code/evaluate_external.py --seed $3 --graph_path "$2" \
    --checkpoint "$1/e2e/trained_model.pth" --output "$1/eval_external_matched.json" > "$1/eval_external_matched.log" 2>&1 &&
  { [ "${4:-}" != lead ] || python code/lead_prioritization.py --seed $3 --run_dir "$1" --graph_path "$2" > "$1/lead_prioritization.log" 2>&1; }
}
phase3() {  # $1 run dir, $2 graph, $3 seed, $4 Phase 2 checkpoint, $5 extra Phase 3a flags
  python -u code/train_classifier.py --seed $3 --results_dir "$1/head" --graph_path "$2" --neg_ratio 1 \
    --phase2_checkpoint "$4" $PM ${5:-} > "$1/head.log" 2>&1 &&
  python -u code/train_end_to_end.py --seed $3 --results_dir "$1/e2e" --graph_path "$2" --neg_ratio 1 \
    --init_checkpoint "$1/head/trained_model.pth" $PM > "$1/e2e.log" 2>&1
}
for s in "${SEEDS[@]}"; do
  G=$(graph $s); bb=results/backbones/backbone_seed_$s
  # Without soft edges: Phase 2 continued from the same Phase 1 encoder with the soft-edge module disabled.
  a=$R/ablation_no_softedge_seed_$s; ab=results/backbones/ablation_no_softedge_seed_$s/backbone; mkdir -p "$a" "$ab"
  python -u code/train_backbone_no_soft_edges.py --seed $s --results_dir "$ab" --graph_path "$G" \
    --num_epochs_phase1 200 --num_epochs_phase2 200 --resume_phase2 --phase1_checkpoint "$bb/best_model_phase1.pth" \
    > "$ab/train.log" 2>&1 && phase3 "$a" "$G" $s "$ab/best_model_phase2.pth" && evaluate "$a" "$G" $s lead
  echo "no-soft-edge seed $s exit=$?"
  # Descriptor-only: graph term of the fusion disabled.
  d=$R/descriptor_only_seed_$s; mkdir -p "$d"
  phase3 "$d" "$G" $s "$bb/best_model_phase2.pth" --descriptor_only && evaluate "$d" "$G" $s lead
  echo "descriptor-only seed $s exit=$?"
  # Random augmentation: same number, type, and weight of added edges, chosen uniformly at random.
  r=$R/random_aug_seed_$s; mkdir -p "$r/backbone"
  python -u code/train_backbone_random_edges.py --seed $s --results_dir "$r/backbone" --graph_path "$G" \
    --num_epochs_phase1 200 --num_epochs_phase2 200 --resume_phase2 --phase1_checkpoint "$bb/best_model_phase1.pth" \
    --phase2_attention_batch_size 100 > "$r/backbone/train.log" 2>&1 &&
  phase3 "$r" "$G" $s "$r/backbone/best_model_phase2.pth" && evaluate "$r" "$G" $s lead
  echo "random-augmentation seed $s exit=$?"
done
