#!/bin/bash
set -euo pipefail
source scripts/00_setup_env.sh
python -m miqthesis.data.download \
  --benchmark_dataset ml-jku/moleculariq-v0.0 \
  --train_pool_dataset ml-jku/moleculariq-trainPool \
  --out_dir data

