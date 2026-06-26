#!/bin/bash
#SBATCH --job-name=miq_poster_figs
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

# Train-env GPU stages (checkpoint sweep + validation attempts) plus all figures.
# Protocol A (#2/#3/#5) is left off here because it runs in the vLLM eval env with
# its own walltime — submit slurm/eval_all.slurm first (its tables are reused
# automatically), or set MIQ_RUN_EVAL=1 below and raise --time accordingly.
export MIQ_RUN_SWEEP=1
export MIQ_RUN_VALIDATION=1
export MIQ_RUN_EVAL="${MIQ_RUN_EVAL:-0}"

bash scripts/10_make_poster_figures.sh
