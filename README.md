# MolecularIQ thesis pipeline

This repository implements a reproducible comparison of full-parameter
multitask SFT and symbolic-verifier GRPO on MolecularIQ with Qwen2.5.

The scientific comparison is intentionally strict:

- Every SFT and GRPO run updates all model parameters. PEFT, LoRA, quantized
  training, frozen backbones, and trainable side modules are rejected.
- Benchmark test molecules are canonicalized with RDKit and removed from the
  training molecule pool before task generation.
- Protocol A is the controlled internal comparison. Decoding, system prompt,
  extractor, and repeats are held fixed across variants.
- Protocol B is a separate leaderboard-comparable run for the selected best
  model and uses the official native settings.
- `avg_accuracy` is the headline metric. Repeat-level pass@1/pass@3 are retained
  with explicit evaluator provenance because the upstream task schema has
  changed over time.
- Model-size escalation is decided from validation metrics only.

## Repository map

- `configs/`: all paths, data sizes, models, training, and evaluation settings.
- `src/miqthesis/data/`: downloads, leakage filtering, symbolic task generation.
- `src/miqthesis/training/`: full-parameter SFT/GRPO and checkpoint invariants.
- `src/miqthesis/evaluation/`: lm-eval wrapper, parsing, verification, aggregation.
- `src/miqthesis/analysis/`: bootstrap tests, tables, plots, report cards.
- `scripts/` and `slurm/`: local orchestration and Leonardo jobs.
- `tests/`: inexpensive acceptance tests that run before GPU jobs.

## Environment

Use Python 3.10 or 3.11. On Leonardo, keep data, caches, environments, and
checkpoints under `$WORK` or `$SCRATCH`, not `$HOME`.

```bash
git clone <this-repository> "$WORK/moleculariq-thesis"
cd "$WORK/moleculariq-thesis"

export MIQ_INSTALL_DEPS=1
source scripts/00_setup_env.sh
unset MIQ_INSTALL_DEPS

git clone https://github.com/ml-jku/moleculariq-eval.git external/moleculariq-eval
git -C external/moleculariq-eval checkout 425ecaaa8faf65aa43aa60ec0f584b7b7f060063
python -m pip install -e "external/moleculariq-eval[vllm]"
python -m pip freeze > requirements-lock.txt
python -m pytest
```

Record the exact `moleculariq-eval` commit. The wrapper also writes it to each
`run_manifest.json`.

## Model staging

Leonardo compute nodes may not have outbound internet access. Download the
required public model snapshots once from a login node:

```bash
bash scripts/01_download_models.sh
```

This creates:

```text
models/Qwen2.5-0.5B/
models/Qwen2.5-0.5B-Instruct/
```

All SLURM jobs force Hugging Face, Transformers, Datasets, and W&B into offline
mode. A missing snapshot therefore fails immediately with the staging command
instead of repeatedly attempting network requests.

## Data preparation

```bash
bash scripts/01_download_data.sh
sbatch slurm/prepare_data.slurm
```

This produces `data/processed/leakage_report.json`, filtered molecules, SFT
JSONL files, and prompt-only GRPO files. The training pool contains SMILES, so
questions and exact targets are generated with `moleculariq-core` after leakage
filtering. The corpus build is streamed to disk, but it remains CPU-intensive
and must run as a SLURM job rather than on a Leonardo login node. The provided
job uses a self-chaining sequence of 18 jobs on the free `lrd_all_serial`
partition. Only one shard is submitted at a time, avoiding the partition's
per-user submitted-job limit. Each shard stays within its four-core, roughly
30 GB RAM, four-hour limits; the final shard submits the assembly/GRPO job.

## Training

The SLURM jobs are configured for account `EUHPC_D27_069`. Submit:

```bash
test -f models/Qwen2.5-0.5B/config.json
sbatch slurm/sft_multitask_debug.slurm
sbatch slurm/sft_count.slurm
sbatch slurm/sft_index.slurm
sbatch slurm/sft_generation.slurm
sbatch slurm/sft_multitask.slurm
```

After SFT:

```bash
bash scripts/05_prepare_checkpoints.sh
sbatch slurm/eval_validation.slurm
sbatch slurm/grpo_format.slurm
sbatch slurm/grpo_verifier.slurm
```

Inspect `results/validation/model_size_decision.json` before starting GRPO. If
the pre-registered trigger fires, rerun the complete SFT variant set at the next
size before proceeding. Run checkpoint preparation again after GRPO. It writes byte-identical
`generation_config.json` files and fails if generation config or chat templates
differ across Protocol A checkpoints.

## Evaluation protocols

Protocol A, all variants:

```bash
bash scripts/06_eval_model.sh A
bash scripts/06_eval_model.sh D  # deterministic, one repeat, analysis-only
bash scripts/07_collect_results.sh
```

After choosing the best model from Protocol A validation results, Protocol B:

```bash
bash scripts/06_eval_model.sh B grpo_verifier
```

Protocol A and B outputs are stored under separate directories and must never
be pooled in a cross-model statistical comparison. Protocol B resolves
`leaderboard_model_path` from `configs/models.yaml`, so it uses the original
checkpoint's native generation settings rather than the controlled Protocol A
copy.

## Pre-registered size escalation

Prepare a CSV containing only validation rows with `model_id`,
`avg_accuracy`, and `split=validation`, then run:

```bash
python -m miqthesis.analysis.model_selection \
  --validation_metrics results/tables/validation_metrics.csv \
  --current_model Qwen/Qwen2.5-0.5B \
  --margin_points 2.0 \
  --output results/tables/model_size_decision.json
```

The ladder is 0.5B -> 1.5B -> 3B. If escalation triggers, rerun the complete
variant matrix at the selected size; do not compare mixed model sizes.
The training CLIs accept `--model_name_or_path`, `--output_dir`, and `--run_id`
overrides so the same registered configs can be materialized at the next size.

## Analysis

```bash
bash scripts/08_make_plots.sh
python -m miqthesis.analysis.report_card \
  --model_id grpo_verifier
```

The aggregation is repeat-aware: `avg_accuracy` averages attempts, whereas
pass@1/pass@3 are computed once per benchmark item. Statistical tests use
Protocol A item-level paired data only.

## Upstream metric provenance

The implementation plan described a MolecularIQ task exposing only
`avg_accuracy` plus bypassed extracted answers. The upstream main branch
inspected during implementation (April 15, 2026 commit history) also exposes
native `pass_at_1` and `pass_at_3`. To keep the thesis reproducible:

1. pin the eval repository commit used for final runs;
2. treat `avg_accuracy` as the headline metric regardless;
3. store native metrics and self-derived repeat metrics separately in parsed
   metadata;
4. do not compare runs produced by different evaluator commits without an
   explicit compatibility analysis.
