#!/usr/bin/env bash
# Convenience wrapper (§8 repo layout): sweep every dataset currently in
# the registry (src/data/loaders.py:REGISTRY) through the full pipeline,
# then regenerate tables and figures. Extend the DATASETS list as more
# entries are added to the registry (§9 week 8: scale to all 12).
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS=("adult" "german_credit" "electricity")
N_SEEDS="${N_SEEDS:-5}"   # plan calls for 10 (§7); default lower here for speed

for ds in "${DATASETS[@]}"; do
  echo "=== running $ds (n_seeds=$N_SEEDS) ==="
  python scripts/run_all.py dataset="$ds" signals=tier_a_c experiment.n_seeds="$N_SEEDS"
done

python scripts/make_tables.py
python scripts/make_figures.py
