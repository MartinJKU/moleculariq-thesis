#!/bin/bash
#SBATCH --job-name=miq_prepare_data
#SBATCH --account=EUHPC_D27_069
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=tmpfs:20G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$WORK/moleculariq-thesis"
source scripts/00_setup_env.sh
bash scripts/02_prepare_sft.sh
