#!/bin/bash
set -euo pipefail
source scripts/00_setup_env.sh
VARIANT="${1:-verifier_reward}"
python -m miqthesis.training.train_grpo --config "configs/grpo/qwen05_${VARIANT}.yaml"

