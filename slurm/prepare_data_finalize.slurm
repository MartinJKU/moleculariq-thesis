#!/bin/bash
#SBATCH --job-name=miq_prepare_finalize
#SBATCH --account=EUHPC_D27_069
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gres=tmpfs:10G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$WORK/moleculariq-thesis"
source scripts/00_setup_env.sh
python -m miqthesis.data.prepare_sft \
  --config configs/data.yaml \
  --stage finalize
python -m miqthesis.data.prepare_grpo --config configs/data.yaml
