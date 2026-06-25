#!/bin/bash
set -euo pipefail

# MolecularIQ uses two virtual environments because the training and evaluation
# stacks have mutually incompatible pins: TRL 0.23 needs Transformers >= 4.56,
# while vLLM 0.6.4 (the last vLLM built for Leonardo's torch 2.5.1) only loads
# Transformers ~4.46. Each environment is internally consistent, so there is no
# repair step. Select with MIQ_ENV=train (default) or MIQ_ENV=eval.
MIQ_ENV="${MIQ_ENV:-train}"

if command -v module >/dev/null 2>&1; then
  module load python/3.11 2>/dev/null || true
  module load cuda 2>/dev/null || true
fi

export MIQ_PROJECT_ROOT="${MIQ_PROJECT_ROOT:-$PWD}"
export HF_HOME="${HF_HOME:-${WORK:-$PWD}/.cache/huggingface}"
export WANDB_PROJECT="${WANDB_PROJECT:-moleculariq-thesis}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -n "${SLURM_JOB_ID:-}" && "${MIQ_ALLOW_COMPUTE_NETWORK:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export DATASETS_OFFLINE=1
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

case "$MIQ_ENV" in
  train) MIQ_DEFAULT_VENV="${WORK:-$PWD}/venvs/moleculariq-thesis" ;;
  eval)  MIQ_DEFAULT_VENV="${WORK:-$PWD}/venvs/moleculariq-eval" ;;
  *)
    echo "Unknown MIQ_ENV='$MIQ_ENV' (expected 'train' or 'eval')." >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

VENV_DIR="${MIQ_VENV_DIR:-$MIQ_DEFAULT_VENV}"
if [[ ! -d "$VENV_DIR" ]]; then
  python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

if [[ "${MIQ_INSTALL_DEPS:-0}" == "1" ]]; then
  python -m pip install --upgrade pip
  # Torch 2.5.1's CUDA 12.1 wheel loads on Leonardo's CUDA 12.2 driver through
  # CUDA minor-version compatibility. Install it first, from the CUDA wheel
  # index, so nothing else selects a different CUDA build.
  python -m pip install \
    "torch==2.5.1" "torchvision==0.20.1" \
    --index-url https://download.pytorch.org/whl/cu121
  if [[ "$MIQ_ENV" == "train" ]]; then
    python -m pip install -e ".[train,chem,dev]"
  else
    if [[ ! -d external/moleculariq-eval ]]; then
      echo "external/moleculariq-eval is missing; clone it before building the eval env." >&2
      return 1 2>/dev/null || exit 1
    fi
    # One resolution pass under constraints-eval.txt keeps Transformers in
    # vLLM's compatible window so the lm-eval harness cannot drag in a release
    # that vLLM 0.6.4 fails to import.
    python -m pip install -c constraints-eval.txt \
      -e . \
      "vllm==0.6.4.post1" \
      -e "external/moleculariq-eval[vllm]"
  fi
  python -m pip check
fi

mkdir -p logs results/raw results/parsed results/tables results/plots results/report_cards
