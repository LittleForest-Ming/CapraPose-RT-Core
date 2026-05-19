#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/caprapose_rt_decoder.py "$@"
