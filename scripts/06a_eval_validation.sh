#!/bin/bash
set -euo pipefail
source scripts/00_setup_env.sh
python -m miqthesis.evaluation.run_validation \
  --config configs/eval/validation.yaml \
  --models configs/models.yaml \
  --model_id instruct_qwen05 \
  --model_id sft_multitask
python -m miqthesis.analysis.model_selection \
  --validation_metrics results/validation/validation_metrics.csv \
  --current_model "${MIQ_CURRENT_MODEL:-Qwen/Qwen2.5-0.5B}" \
  --margin_points "${MIQ_ESCALATION_MARGIN:-2.0}" \
  --output results/validation/model_size_decision.json

