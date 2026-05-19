#!/usr/bin/env bash
set -euo pipefail

python tools/export_experiment_qualitative.py \
  --config configs/caprapose_rt.py \
  --prediction-json outputs/experiments/caprapose_rt_full_goat17/evaluation/val_best/predictions.json \
  --split val \
  "$@"
