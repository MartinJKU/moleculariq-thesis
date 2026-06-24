#!/bin/bash
set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load python/3.11 2>/dev/null || true
  module load cuda 2>/dev/null || true
fi

export MIQ_PROJECT_ROOT="${MIQ_PROJECT_ROOT:-$PWD}"
export HF_HOME="${HF_HOME:-${WORK:-$PWD}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export WANDB_PROJECT="${WANDB_PROJECT:-moleculariq-thesis}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -n "${SLURM_JOB_ID:-}" && "${MIQ_ALLOW_COMPUTE_NETWORK:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export DATASETS_OFFLINE=1
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

VENV_DIR="${MIQ_VENV_DIR:-${WORK:-$PWD}/venvs/moleculariq-thesis}"
if [[ ! -d "$VENV_DIR" ]]; then
  python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

if [[ "${MIQ_INSTALL_DEPS:-0}" == "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -e ".[train,chem]"
fi

mkdir -p logs results/raw results/parsed results/tables results/plots results/report_cards
