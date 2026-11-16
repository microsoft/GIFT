# Sourced by the pipeline scripts: run from the repository root, seeds 42-71 unless given on the command line.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=($(seq 42 71))
graph() { echo "data/graphs/graph_seed$1.pkl"; }
PM="--iso_fraction 0.30 --protein_mask_fraction 0.30"   # Phase 3 compound and protein masking (30%)
