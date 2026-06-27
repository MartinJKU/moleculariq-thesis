#!/bin/bash
#SBATCH --job-name=miq_eval_grpo
#SBATCH --account=EUHPC_D27_069
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$WORK/moleculariq-thesis"
source scripts/00_setup_env.sh
# Protocol A for only the listed models, then re-collect. Use this to re-run the
# models that failed without redoing base/instruct/SFT. Override the list with
# MIQ_EVAL_MODEL_IDS if needed.
export MIQ_EVAL_MODEL_IDS="${MIQ_EVAL_MODEL_IDS:-grpo_format grpo_verifier}"
bash scripts/06_eval_model.sh A
bash scripts/07_collect_results.sh
