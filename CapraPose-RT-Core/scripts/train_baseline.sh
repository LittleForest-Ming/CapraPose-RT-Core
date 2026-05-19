#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/baseline_rtmpose.py "$@"
