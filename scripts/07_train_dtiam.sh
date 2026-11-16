#!/usr/bin/env bash
# DTIAM baseline (Nat. Commun. 2025) under the shared protocol: frozen BerMol + ESM-2 encoders, AutoGluon head per run.
# Needs: DTIAM_DIR (official code, commit 61d68b8, with BerMolModel_base.pkl), DTIAM_PY (Python 3.9 + AutoGluon 0.5.2,
# requirements-dtiam.txt), fair-esm in the main environment, GUACAMOL_SMILES for the pretraining-overlap audit.
source "$(dirname "$0")/common.sh"
: "${DTIAM_DIR:?set DTIAM_DIR to the official DTIAM checkout}"; : "${DTIAM_PY:?set DTIAM_PY to the AutoGluon Python}"
R=results/robustness/dtiam
for s in "${SEEDS[@]}"; do
  mkdir -p $R/runs/run_$s
  [ -f $R/runs/run_$s/protocol.pkl ] || python code/train_dta_baselines.py --seed $s --graph_path "$(graph $s)" \
    --output $R/runs/run_$s/unused.json --export_protocol $R/runs/run_$s/protocol.pkl
done
[ -f data/dtiam/dtiam_features.pkl ] || python code/robustness/dtiam_featurize.py
for s in "${SEEDS[@]}"; do
  d=$R/runs/run_$s
  PYTHONPATH=. "$DTIAM_PY" code/robustness/dtiam_train.py --protocol $d/protocol.pkl --features data/dtiam/dtiam_features.pkl \
    --scores_out $d/scores.npz --lead_out $d/lead_dtiam.json --workdir $d/ag --num_cpus "${NUM_CPUS:-8}" --keep_models > $d/train.log 2>&1 &&
  python code/train_dta_baselines.py --seed $s --graph_path "$(graph $s)" --import_scores $d/scores.npz \
    --output $d/dta_baselines_dtiam.json --scores_out $d/dta_baselines_dtiam_scores.npz > $d/eval.log 2>&1
  echo "dtiam seed $s done"
done
[ -n "${GUACAMOL_SMILES:-}" ] && python code/robustness/dtiam_pretraining_overlap.py   # overlap of BerMol pretraining data with the 95 drugs
python code/robustness/dtiam_numbers.py
