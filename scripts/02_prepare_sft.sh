#!/bin/bash
set -euo pipefail
source scripts/00_setup_env.sh
python -m miqthesis.data.prepare_sft --config configs/data.yaml
python -m miqthesis.data.prepare_grpo --config configs/data.yaml

