#!/bin/bash
#SBATCH --job-name=miq_grpo_verifier
#SBATCH --account=<PROJECT_ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$WORK/moleculariq-thesis"
source scripts/00_setup_env.sh
python -m miqthesis.training.train_grpo --config configs/grpo/qwen05_verifier_reward.yaml

