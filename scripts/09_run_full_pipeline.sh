#!/bin/bash
set -euo pipefail
source scripts/00_setup_env.sh
bash scripts/01_download_data.sh
bash scripts/02_prepare_sft.sh
for variant in count index generation multitask; do
  bash scripts/03_train_sft.sh "$variant"
done
bash scripts/05_prepare_checkpoints.sh
bash scripts/06a_eval_validation.sh
if ! python -c "import json,sys; d=json.load(open('results/validation/model_size_decision.json')); sys.exit(0 if not d['escalate'] else 1)"; then
  echo "Model-size escalation triggered. Re-run the full variant set at the next registered size." >&2
  exit 3
fi
bash scripts/04_train_grpo.sh format_reward
bash scripts/04_train_grpo.sh verifier_reward
bash scripts/05_prepare_checkpoints.sh
bash scripts/06_eval_model.sh A
bash scripts/07_collect_results.sh
bash scripts/08_make_plots.sh
