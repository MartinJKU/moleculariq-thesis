#!/bin/bash
#SBATCH --job-name=miq_sweep_ckpts
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
# Light HF-generate validation path -> runs in the training env (the default).
source scripts/00_setup_env.sh

# Evaluate every saved checkpoint of the GRPO runs on a fixed validation subset.
# Override the run list / baseline / subset size with env vars if needed.
MIQ_SWEEP_RUNS="${MIQ_SWEEP_RUNS:-verifier=checkpoints/grpo_verifier format=checkpoints/grpo_format}"
MIQ_SWEEP_BASELINE="${MIQ_SWEEP_BASELINE:-sft=checkpoints/eval/sft_multitask}"
MIQ_SWEEP_LIMIT="${MIQ_SWEEP_LIMIT:-400}"

run_args=()
for spec in $MIQ_SWEEP_RUNS; do
  run_args+=(--run "$spec")
done

python -m miqthesis.evaluation.sweep_checkpoints \
  "${run_args[@]}" \
  --baseline "$MIQ_SWEEP_BASELINE" \
  --limit "$MIQ_SWEEP_LIMIT" \
  --output results/validation/checkpoint_sweep.csv

# Render the learning-curve figure (and the other poster-only plots if present).
python -m miqthesis.analysis.poster_plots \
  --sweep_csv results/validation/checkpoint_sweep.csv \
  --output_dir results/plots
