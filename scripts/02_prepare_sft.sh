#!/bin/bash
set -euo pipefail

if [[ "$(hostname -s)" == login* && -z "${SLURM_JOB_ID:-}" && "${MIQ_ALLOW_LOGIN_PREP:-0}" != "1" ]]; then
  echo "Refusing the full corpus build on a Leonardo login node." >&2
  echo "Submit slurm/prepare_data.slurm with sbatch instead." >&2
  echo "Set MIQ_ALLOW_LOGIN_PREP=1 only for an intentionally tiny debug config." >&2
  exit 2
fi

source scripts/00_setup_env.sh
python -m miqthesis.data.prepare_sft --config configs/data.yaml
python -m miqthesis.data.prepare_grpo --config configs/data.yaml
