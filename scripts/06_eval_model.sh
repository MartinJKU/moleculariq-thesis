#!/bin/bash
set -euo pipefail
# lm-eval runs on the vLLM stack, which lives in its own environment.
export MIQ_ENV=eval
source scripts/00_setup_env.sh
PROTOCOL="${1:-A}"
MODEL_ID="${2:-}"
if [[ "$PROTOCOL" == "A" ]]; then
  args=(--config configs/eval/moleculariq_pass_at_k.yaml --models configs/models.yaml)
elif [[ "$PROTOCOL" == "D" || "$PROTOCOL" == "A_DETERMINISTIC" ]]; then
  args=(--config configs/eval/moleculariq_deterministic.yaml --models configs/models.yaml)
else
  if [[ -z "$MODEL_ID" ]]; then
    echo "Protocol B requires the selected best model id." >&2
    exit 2
  fi
  args=(--config configs/eval/leaderboard.yaml --models configs/models.yaml --model_id "$MODEL_ID")
fi
# Optionally restrict Protocol A/D to specific models (space-separated ids), so a
# re-run can target just the models that failed without redoing the whole matrix.
if [[ -n "${MIQ_EVAL_MODEL_IDS:-}" ]]; then
  for mid in $MIQ_EVAL_MODEL_IDS; do args+=(--model_id "$mid"); done
fi
python -m miqthesis.evaluation.run_lm_eval "${args[@]}" \
  --eval_repo "${MOLECULARIQ_EVAL_DIR:-external/moleculariq-eval}"
