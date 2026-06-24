#!/bin/bash
#SBATCH --job-name=miq_prepare_data
#SBATCH --account=EUHPC_D27_069
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --gres=tmpfs:10G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$WORK/moleculariq-thesis"
source scripts/00_setup_env.sh

shard_index="${MIQ_SHARD_INDEX:-0}"
shard_count="$(
  python -m miqthesis.data.prepare_sft \
    --config configs/data.yaml \
    --print-shard-count
)"

echo "Preparing shard $shard_index of $((shard_count - 1))"
python -m miqthesis.data.prepare_sft \
  --config configs/data.yaml \
  --stage shard \
  --task-index "$shard_index"

next_index=$((shard_index + 1))
if (( next_index < shard_count )); then
  next_job="$(
    sbatch --parsable \
      --export=ALL,MIQ_SHARD_INDEX="$next_index" \
      slurm/prepare_data.slurm
  )"
  echo "Submitted next preparation shard as job $next_job"
else
  final_job="$(sbatch --parsable slurm/prepare_data_finalize.slurm)"
  echo "All shards complete; submitted finalization as job $final_job"
fi
