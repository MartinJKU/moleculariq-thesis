#!/bin/bash
set -euo pipefail
source scripts/00_setup_env.sh
python -m miqthesis.evaluation.parse_lm_eval_outputs \
  --input_dir results/raw/lm_eval \
  --benchmark data/processed/benchmark_test.parquet \
  --output_dir results/parsed
python -m miqthesis.evaluation.aggregate_metrics \
  --input_dir results/parsed \
  --output_dir results/tables

