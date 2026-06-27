#!/bin/bash
# Stage all SFT + GRPO checkpoints into checkpoints/eval/ for evaluation.
#
# Use this instead of calling scripts/05_prepare_checkpoints.sh directly when a
# GRPO run stopped mid-schedule (e.g. at step 2750). In that case the trainer's
# final top-level save_model may not have landed, so prepare_checkpoint would
# fail with "No full model weights found". This script first promotes the latest
# checkpoint-N's inference files (weights + config + tokenizer only — no
# optimizer/scheduler state) to the run's top level, which is a no-op if a final
# model is already there, and then runs the normal prepare + consistency check.
#
# Override the GRPO run list if needed:
#   MIQ_GRPO_RUNS="grpo_format grpo_verifier" bash scripts/05a_stage_eval_checkpoints.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MIQ_GRPO_RUNS="${MIQ_GRPO_RUNS:-grpo_format grpo_verifier}"

# Files required to load a model for evaluation. We copy only these so promoted
# top-level dirs (and their eval copies) do not carry gigabytes of optimizer state.
PROMOTE_GLOBS=(
  config.json generation_config.json
  model.safetensors model.safetensors.index.json "model-*.safetensors"
  pytorch_model.bin pytorch_model.bin.index.json "pytorch_model-*.bin"
  tokenizer.json tokenizer_config.json tokenizer.model
  vocab.json merges.txt special_tokens_map.json added_tokens.json
  chat_template.jinja chat_template.json
)

has_weights() {
  # True if either a safetensors or a bin weight file is present. Each glob is
  # tested independently: `ls a* b*` would return nonzero whenever only one of
  # the two patterns matches, which is the normal case.
  compgen -G "$1/model*.safetensors" >/dev/null 2>&1 && return 0
  compgen -G "$1/pytorch_model*.bin" >/dev/null 2>&1 && return 0
  return 1
}

promote_run() {
  local run="$1" dir="checkpoints/$run"
  if [[ ! -d "$dir" ]]; then
    echo "  - $run: no directory ($dir), skipping"
    return 0
  fi
  if has_weights "$dir"; then
    echo "  - $run: final model already at top level, no promotion needed"
    return 0
  fi
  local latest
  latest="$(ls -d "$dir"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: $run has neither a top-level model nor any checkpoint-* dir in $dir" >&2
    return 1
  fi
  echo "  - $run: promoting inference files from $(basename "$latest") -> $dir/"
  local pat f copied=0
  for pat in "${PROMOTE_GLOBS[@]}"; do
    for f in "$latest"/$pat; do
      [[ -e "$f" ]] || continue   # skip non-matching globs / absent optional files
      cp -f "$f" "$dir/"
      copied=$((copied + 1))
    done
  done
  if ! has_weights "$dir"; then
    echo "ERROR: promotion produced no model weights in $dir (source $latest)" >&2
    return 1
  fi
  echo "    promoted $copied files"
}

echo "==> Promoting truncated GRPO runs if their final save is missing"
for run in $MIQ_GRPO_RUNS; do
  promote_run "$run"
done

echo "==> Staging all checkpoints into checkpoints/eval/ (prepare + consistency validate)"
bash scripts/05_prepare_checkpoints.sh

echo "==> Done. Eval-ready checkpoints under checkpoints/eval/:"
if [[ -d checkpoints/eval ]]; then
  ls -1 checkpoints/eval | sed 's/^/    /'
else
  echo "    (none created — check the log above)"
fi
