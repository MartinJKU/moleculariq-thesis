#!/bin/bash
# Ensure every staged eval checkpoint has a chat_template.jinja so Protocol A
# (lm-eval / vLLM) can apply the chat template.
#
# Transformers 4.56 stores the chat template in a standalone chat_template.jinja
# file. A checkpoint staged from an intermediate GRPO step can be missing it,
# which makes lm-eval fail at apply_chat_template even though the model loads and
# passes the eos-token guard (the consistency check reads tokenizer_config.json,
# where the template is empty for everyone, so it does not catch this).
#
# All variants share the identical Qwen tokenizer and chat template, so this
# copies the proven-good tokenizer files from a reference eval checkpoint
# (sft_multitask by default) into any eval dir that lacks the template, then
# re-validates. Idempotent: a no-op once every checkpoint has the template.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EVAL_ROOT="${MIQ_EVAL_ROOT:-checkpoints/eval}"
if [[ ! -d "$EVAL_ROOT" ]]; then
  echo "No $EVAL_ROOT directory; stage checkpoints first (scripts/05a_stage_eval_checkpoints.sh)." >&2
  exit 1
fi

# Tokenizer files are identical across Qwen variants; copy the whole set so a
# partially-staged checkpoint is fully repaired, not just the template file.
TOKENIZER_FILES=(
  chat_template.jinja chat_template.json
  tokenizer_config.json tokenizer.json tokenizer.model
  vocab.json merges.txt special_tokens_map.json added_tokens.json
)

# Reference = a checkpoint that already has a chat template (prefer sft_multitask).
refdir=""
if [[ -f "$EVAL_ROOT/sft_multitask/chat_template.jinja" ]]; then
  refdir="$EVAL_ROOT/sft_multitask"
else
  for d in "$EVAL_ROOT"/*/; do
    if [[ -f "${d%/}/chat_template.jinja" ]]; then refdir="${d%/}"; break; fi
  done
fi
if [[ -z "$refdir" ]]; then
  echo "No chat_template.jinja found under $EVAL_ROOT/* — nothing to copy from."
  echo "(If your tokenizers embed the template in tokenizer_config.json instead, this is expected.)"
  exit 0
fi
echo "Reference checkpoint with chat template: $refdir"

fixed=0
for d in "$EVAL_ROOT"/*/; do
  d="${d%/}"
  [[ "$d" == "$refdir" ]] && continue
  if [[ ! -f "$d/chat_template.jinja" ]]; then
    for f in "${TOKENIZER_FILES[@]}"; do
      [[ -f "$refdir/$f" ]] && cp -f "$refdir/$f" "$d/"
    done
    echo "  + repaired tokenizer (incl. chat template) in $d"
    fixed=$((fixed + 1))
  fi
done
echo "Repaired $fixed checkpoint(s)."

# Re-validate the controlled set (eos guard + chat-template consistency).
# shellcheck disable=SC1091
source scripts/00_setup_env.sh
python -m miqthesis.training.prepare_checkpoint --validate_only --eval_root "$EVAL_ROOT"
echo "==> chat-template repair complete."
