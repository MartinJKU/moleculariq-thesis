#!/bin/bash
set -euo pipefail
source scripts/00_setup_env.sh
python -m miqthesis.analysis.statistical_tests \
  --input_dir results/parsed \
  --output results/tables/statistical_tests.csv
python -m miqthesis.analysis.make_tables \
  --input_dir results/tables \
  --output_dir results/tables
python -m miqthesis.analysis.make_plots \
  --tables_dir results/tables \
  --parsed_dir results/parsed \
  --output_dir results/plots

