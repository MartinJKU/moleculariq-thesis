#!/bin/bash
#SBATCH --job-name=miq_eval_leaderboard
#SBATCH --account=EUHPC_D27_069
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$WORK/moleculariq-thesis"
# Protocol B is best-model-only. Pass the selected model id at submission time:
#   sbatch --export=ALL,MIQ_LEADERBOARD_MODEL=grpo_verifier slurm/eval_leaderboard.slurm
: "${MIQ_LEADERBOARD_MODEL:?Set MIQ_LEADERBOARD_MODEL to the selected best model id, e.g. grpo_verifier}"
source scripts/00_setup_env.sh
bash scripts/06_eval_model.sh B "$MIQ_LEADERBOARD_MODEL"
