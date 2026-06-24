#!/bin/bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Model snapshots must be downloaded on a network-enabled login node." >&2
  exit 2
fi

source scripts/00_setup_env.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE DATASETS_OFFLINE
python -m miqthesis.data.download_models --model-root models

