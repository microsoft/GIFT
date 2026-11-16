#!/usr/bin/env bash
# Rebuild the 30 per-run knowledge graphs (median-pChEMBL labels, random 70/15/15 split per seed) from the base
# graph data/graphs/graph_base.pkl and the ChEMBL 36 SQLite release (CHEMBL36_DB).
# The released graphs in data/graphs/ are the output of this step; rerunning it is optional.
# Usage: CHEMBL36_DB=/path/to/chembl_36.db scripts/01_build_graphs.sh [SEED ...]
source "$(dirname "$0")/common.sh"
: "${CHEMBL36_DB:?set CHEMBL36_DB to the ChEMBL 36 SQLite file}"
python code/build_run_graphs.py --chembl_db "$CHEMBL36_DB" --split_seeds "$(IFS=,; echo "${SEEDS[*]}")"
