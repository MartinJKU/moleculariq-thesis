#!/bin/bash
set -euo pipefail
source scripts/00_setup_env.sh
for variant in sft_count sft_index sft_generation sft_multitask grpo_format grpo_verifier grpo_verifier_ablation; do
  if [[ -d "checkpoints/$variant" ]]; then
    python -m miqthesis.training.prepare_checkpoint \
      --source "checkpoints/$variant" \
      --destination "checkpoints/eval/$variant" \
      --eval_root checkpoints/eval
  fi
done
python -m miqthesis.training.prepare_checkpoint \
  --validate_only \
  --eval_root checkpoints/eval

