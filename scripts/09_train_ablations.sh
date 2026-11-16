#!/usr/bin/env bash
# Ablations of Phase 3 node masking and of contrastive pretraining (Phases 1-2), otherwise identical to scripts/02.
#   no_mask     : pretrained Phase-2 encoder, Phase 3a/3b without masking
#   no_pretrain : randomly initialized encoder (frozen in 3a), Phase 3 masking unchanged
# LR (Phase-3a learning rate) defaults to the main model's 1e-3. Usage: 09_train_ablations.sh [seeds]
source "$(dirname "$0")/common.sh"
LR=${LR:-1e-3}; R=results/robustness/ablation_phases
for v in no_mask no_pretrain; do
  for s in "${SEEDS[@]}"; do
    G=$(graph $s); d=$R/$v/seed_$s; BB=results/backbones/backbone_seed_$s/best_model_phase2.pth; M="$PM"
    mkdir -p "$d"
    [ $v = no_mask ] && M="--iso_fraction 0 --protein_mask_fraction 0"
    if [ $v = no_pretrain ]; then
      mkdir -p $R/random_init
      [ -f $R/random_init/seed_$s.pth ] || python code/robustness/ablation_random_init.py --seed $s --graph_path "$G" \
        --out $R/random_init/seed_$s.pth --reference "$BB" > $R/random_init/seed_$s.log 2>&1
      BB=$R/random_init/seed_$s.pth
    fi
    python -u code/train_classifier.py --seed $s --results_dir "$d/head" --graph_path "$G" --neg_ratio 1 \
      --phase2_checkpoint "$BB" $M --lr $LR > "$d/head.log" 2>&1 &&
    python -u code/train_end_to_end.py --seed $s --results_dir "$d/e2e" --graph_path "$G" --neg_ratio 1 \
      --init_checkpoint "$d/head/trained_model.pth" $M > "$d/e2e.log" 2>&1 &&
    python code/robustness/ablation_evaluate.py --ckpt "$d/e2e/trained_model.pth" --seed $s --out "$d/eval_fast.json" > "$d/eval_fast.log" 2>&1
    echo "ablation $v seed $s done"
  done
done
python code/robustness/ablation_main_comparison.py
