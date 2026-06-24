#!/bin/bash
set -euo pipefail
source scripts/00_setup_env.sh
VARIANT="${1:-multitask}"
python -m miqthesis.training.train_sft --config "configs/sft/qwen05_${VARIANT}.yaml"

