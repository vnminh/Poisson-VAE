#!/usr/bin/env bash
# Full grid of the report for one dataset: 3 losses x 7 betas x 4 seeds, then evaluation.
#   bash scripts/run_grid.sh nslkdd
set -e
DATASET=${1:?usage: run_grid.sh <nslkdd|cicids2017>}
for LOSS in mse poisson negbin; do
  for BETA in 0 1e-5 1e-4 1e-3 1e-2 0.1 0.5; do
    for SEED in 0 1 2 3; do
      python scripts/train.py --dataset "$DATASET" --loss "$LOSS" --beta "$BETA" --seed "$SEED"
    done
  done
done
python scripts/eval.py --dataset "$DATASET"
