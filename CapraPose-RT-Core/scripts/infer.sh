#!/usr/bin/env bash
set -euo pipefail

python infer.py \
  --config configs/caprapose_rt.py \
  --checkpoint outputs/experiments/caprapose_rt_full_goat17/checkpoints/best.pth \
  --image path/to/goat_crop.jpg \
  --output outputs/prediction.json \
  --save-vis outputs/prediction_vis.jpg \
  "$@"
