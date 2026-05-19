#!/usr/bin/env bash
set -euo pipefail

python eval.py \
  --config configs/caprapose_rt.py \
  --checkpoint outputs/experiments/caprapose_rt_full_goat17/checkpoints/best.pth \
  "$@"
