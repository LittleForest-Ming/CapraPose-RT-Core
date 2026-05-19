#!/usr/bin/env bash
set -euo pipefail

python tools/benchmark_input_pipeline.py --config configs/caprapose_rt.py --split train --mode train --num-batches 20 "$@"
