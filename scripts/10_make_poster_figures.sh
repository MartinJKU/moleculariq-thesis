#!/bin/bash
# Regenerate all seven poster figures from one entry point.
#
# The figures span two virtual environments and several data artifacts, so this
# script is best-effort: it runs whatever data-generation stages you enable, then
# turns every artifact that exists into a figure and prints an inventory of the
# seven targets at the end. Stages never abort the run — a missing checkpoint or
# absent Protocol A output just leaves its figure un-produced.
#
# Runs in the TRAIN env (default) and shells out to scripts/06_eval_model.sh for
# the Protocol A step, which selects the EVAL (vLLM) env in its own subshell.
#
# Heavy GPU stages are opt-in (default off):
#   MIQ_RUN_EVAL=1        Protocol A lm-eval        -> data for #2, #3, #5
#   MIQ_RUN_SWEEP=1       checkpoint sweep          -> data for #1
#   MIQ_RUN_VALIDATION=1  validation attempts       -> data for #7
#   MIQ_RUN_ALL=1         enable all three
#
# Sweep / qualitative overrides (defaults mirror slurm/sweep_checkpoints.slurm):
#   MIQ_SWEEP_RUNS, MIQ_SWEEP_BASELINE, MIQ_SWEEP_LIMIT, MIQ_QUAL_MODELS, MIQ_VAL_JSONL
#
# Example (everything, on a GPU node):
#   MIQ_RUN_ALL=1 bash scripts/10_make_poster_figures.sh
# Example (figures from data already on disk, no GPU):
#   bash scripts/10_make_poster_figures.sh
set -uo pipefail

MIQ_RUN_EVAL="${MIQ_RUN_EVAL:-0}"
MIQ_RUN_SWEEP="${MIQ_RUN_SWEEP:-0}"
MIQ_RUN_VALIDATION="${MIQ_RUN_VALIDATION:-0}"
if [[ "${MIQ_RUN_ALL:-0}" == "1" ]]; then
  MIQ_RUN_EVAL=1
  MIQ_RUN_SWEEP=1
  MIQ_RUN_VALIDATION=1
fi

MIQ_SWEEP_RUNS="${MIQ_SWEEP_RUNS:-verifier=checkpoints/grpo_verifier format=checkpoints/grpo_format}"
MIQ_SWEEP_BASELINE="${MIQ_SWEEP_BASELINE:-sft=checkpoints/eval/sft_multitask}"
MIQ_SWEEP_LIMIT="${MIQ_SWEEP_LIMIT:-400}"
MIQ_SWEEP_EXTRA="${MIQ_SWEEP_EXTRA:-}"
# Auto-include a checkpoint rescued from rotation (overridable). Space-separated
# entries, each "label=step:path"; the label joins an existing run's curve.
if [[ -z "$MIQ_SWEEP_EXTRA" && -d checkpoints/_attic/grpo_verifier_step1000 ]]; then
  MIQ_SWEEP_EXTRA="verifier=1000:checkpoints/_attic/grpo_verifier_step1000"
fi
MIQ_QUAL_MODELS="${MIQ_QUAL_MODELS:-instruct_qwen05 sft_multitask grpo_verifier}"
MIQ_QUAL_LIMIT="${MIQ_QUAL_LIMIT:-300}"  # #7 needs only a few examples; cap the slow HF-generate pass
MIQ_VAL_JSONL="${MIQ_VAL_JSONL:-data/processed/sft_multitask_val.jsonl}"

PLOTS=results/plots

# shellcheck disable=SC1091
source scripts/00_setup_env.sh   # TRAIN env (default)
mkdir -p "$PLOTS" results/tables results/parsed results/validation

stage() {  # run a stage without letting its failure abort the whole script
  local label="$1"; shift
  echo "::: $label"
  if ! "$@"; then
    echo "WARNING: stage failed and was skipped: $label" >&2
  fi
}

# --- #2,#3,#5 data: Protocol A eval (EVAL env subshell) ---------------------
if [[ "$MIQ_RUN_EVAL" == "1" ]]; then
  stage "Protocol A eval (vLLM)" bash scripts/06_eval_model.sh A
fi

# --- collect + aggregate Protocol A into tables -----------------------------
if compgen -G "results/raw/lm_eval/protocol_a/*" >/dev/null 2>&1; then
  stage "Collect + aggregate Protocol A" bash scripts/07_collect_results.sh
else
  echo "::: skip collect (no results/raw/lm_eval/protocol_a/* — set MIQ_RUN_EVAL=1)"
fi

# --- statistical tests + tables + standard plots (#3, #5, optional) ---------
if compgen -G "results/parsed/*_samples.parquet" >/dev/null 2>&1; then
  stage "Statistical tests" \
    python -m miqthesis.analysis.statistical_tests \
      --input_dir results/parsed --output results/tables/statistical_tests.csv
  stage "LaTeX tables" \
    python -m miqthesis.analysis.make_tables \
      --input_dir results/tables --output_dir results/tables
  stage "Standard make_plots" \
    python -m miqthesis.analysis.make_plots \
      --tables_dir results/tables --parsed_dir results/parsed --output_dir "$PLOTS"
else
  echo "::: skip stats/standard plots (no results/parsed/*_samples.parquet — set MIQ_RUN_EVAL=1)"
fi

# --- #1 data: checkpoint sweep ----------------------------------------------
if [[ "$MIQ_RUN_SWEEP" == "1" ]]; then
  sweep_args=()
  for spec in $MIQ_SWEEP_RUNS; do sweep_args+=(--run "$spec"); done
  for spec in $MIQ_SWEEP_EXTRA; do sweep_args+=(--extra "$spec"); done
  stage "Checkpoint sweep" \
    python -m miqthesis.evaluation.sweep_checkpoints \
      "${sweep_args[@]}" --baseline "$MIQ_SWEEP_BASELINE" \
      --limit "$MIQ_SWEEP_LIMIT" --output results/validation/checkpoint_sweep.csv
fi

# --- #7 data: validation attempts -------------------------------------------
if [[ "$MIQ_RUN_VALIDATION" == "1" ]]; then
  qual_args=()
  for model in $MIQ_QUAL_MODELS; do qual_args+=(--model_id "$model"); done
  stage "Validation attempts" \
    python -m miqthesis.evaluation.run_validation "${qual_args[@]}" --limit "$MIQ_QUAL_LIMIT"
fi

# --- poster-only figures: #1, #2, #4, #6 (whatever inputs exist) ------------
stage "Poster plots (#1/#2/#4/#6)" \
  python -m miqthesis.analysis.poster_plots \
    --sweep_csv results/validation/checkpoint_sweep.csv \
    --stats_csv results/tables/statistical_tests.csv \
    --grpo_logs_dir results/raw/grpo_logs \
    --output_dir "$PLOTS"

# --- #7 qualitative panel ----------------------------------------------------
if compgen -G "results/validation/*_attempts.parquet" >/dev/null 2>&1; then
  stage "Qualitative panel (#7)" \
    python -m miqthesis.analysis.qualitative_panel \
      --attempts_dir results/validation --val_jsonl "$MIQ_VAL_JSONL" --output_dir "$PLOTS"
else
  echo "::: skip qualitative panel (no results/validation/*_attempts.parquet — set MIQ_RUN_VALIDATION=1)"
fi

# --- inventory ---------------------------------------------------------------
echo
echo "==================== poster figure inventory ===================="
have() {
  if [[ -f "$PLOTS/$1.png" ]]; then
    echo "  [x] $2  ->  $PLOTS/$1.png"
  else
    echo "  [ ] $2  (missing $1.png)"
  fi
}
have lineplot_checkpoint_sweep     "#1 GRPO learning curve"
have barplot_leaderboard_ci        "#2 leaderboard bars + 95% CI"
have barplot_task_type_accuracy    "#3 per-task-family bars"
have lineplot_grpo_reward          "#4 training curves"
have scatter_format_vs_correctness "#5 format-vs-correctness scatter"
have lineplot_reward_components    "#6 reward-component decomposition"
have qualitative_panel             "#7 qualitative panel"
echo "================================================================="
