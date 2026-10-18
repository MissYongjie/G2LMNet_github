#!/usr/bin/env bash
# Train G2LMNet on all six HetData pairs and evaluate the final change maps.
# Usage: bash scripts/train_all.sh [DATA_DIR] [WORK_DIR]
set -euo pipefail
DATA_DIR=${1:-data/hetdata}
WORK_DIR=${2:-work_dirs}

for N in 1 2 3 4 5 6; do
  python tools/train.py --config configs/hetdata${N}.yaml --data_dir "$DATA_DIR" --work_dir "$WORK_DIR/hetdata${N}"
done
python tools/evaluate.py --pred_dir "$WORK_DIR" --data_dir "$DATA_DIR" --csv "$WORK_DIR/metrics.csv"
