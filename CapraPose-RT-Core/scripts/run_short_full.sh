#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --config configs/caprapose_rt.py \
  --work-dir outputs/experiments/caprapose_rt_full_goat17/short_run \
  --epochs 2 \
  --train-num-workers 0 \
  --eval-num-workers 0 \
  --max-train-batches 4 \
  --max-eval-batches 2 \
  "$@"
