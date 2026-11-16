#!/usr/bin/env bash
# ESM-2 matched swap: the 453-d protein descriptors are replaced by mean-pooled ESM-2 650M embeddings (needs fair-esm);
# Phases 1-3 and the evaluation are otherwise identical to scripts/02. LR defaults to the main model's 1e-3.
source "$(dirname "$0")/common.sh"
LR=${LR:-1e-3}; R=results/robustness/esm
[ -f data/esm2_650m_embeddings.npz ] || python code/robustness/esm_compute_embeddings.py
python code/robustness/esm_build_graphs.py --seeds "${SEEDS[@]}"
for s in "${SEEDS[@]}"; do
  G=data/graphs/esm/graph_esm_seed$s.pkl; d=$R/run_$s; bb=$d/backbone; mkdir -p "$bb"
  python -u code/train_backbone.py --seed $s --results_dir "$bb" --graph_path "$G" --num_epochs_phase1 200 --num_epochs_phase2 200 \
    --phase2_attention_batch_size 100 > "$bb/train.log" 2>&1 &&
  python -u code/train_classifier.py --seed $s --results_dir "$d/head" --graph_path "$G" --neg_ratio 1 \
    --phase2_checkpoint "$bb/best_model_phase2.pth" $PM --lr $LR > "$d/head.log" 2>&1 &&
  python -u code/train_end_to_end.py --seed $s --results_dir "$d/e2e" --graph_path "$G" --neg_ratio 1 \
    --init_checkpoint "$d/head/trained_model.pth" $PM > "$d/e2e.log" 2>&1 &&
  python code/evaluate_chembl35.py --seed $s --graph_path "$G" --model_checkpoint "$d/e2e/trained_model.pth" --output "$d/eval_final.json" > "$d/eval.log" 2>&1 &&
  python code/evaluate_external.py --seed $s --graph_path "$G" --checkpoint "$d/e2e/trained_model.pth" \
    --output "$d/eval_external_matched.json" > "$d/eval_external_matched.log" 2>&1
  echo "esm seed $s done"
done
python code/robustness/ablation_main_comparison.py
