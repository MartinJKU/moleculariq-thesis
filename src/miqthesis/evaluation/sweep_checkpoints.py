"""Evaluate every saved checkpoint of one or more GRPO runs on a fixed
validation subset, producing the data for the avg_accuracy-vs-step learning
curve. Uses the light HF-generate validation path (training env), NOT the heavy
Protocol A lm-eval/vLLM pipeline, so a full sweep runs in minutes.

The same validation items are reused for every checkpoint (paired comparison),
and each GRPO curve is anchored at step 0 with the SFT init it was trained from.

Example
-------
python -m miqthesis.evaluation.sweep_checkpoints \
  --baseline sft=checkpoints/eval/sft_multitask \
  --run verifier=checkpoints/grpo_verifier \
  --run format=checkpoints/grpo_format \
  --limit 400 \
  --output results/validation/checkpoint_sweep.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from miqthesis.constants import IM_END_TOKEN_ID
from miqthesis.evaluation.run_validation import (
    _load_jsonl,
    evaluate_model,
    validation_items,
)
from miqthesis.training.utils import load_yaml

# <|im_end|> first (real turn-end for chat-trained checkpoints), then <|endoftext|>.
EVAL_EOS = [IM_END_TOKEN_ID, 151643]
_STEP_RE = re.compile(r"checkpoint-(\d+)$")


def _parse_spec(spec: str) -> tuple[str, Path]:
    """Parse a "label=path" (or bare "path") run specification."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, Path(path)
    return Path(spec).name, Path(spec)


def _subsample(
    rows: list[dict[str, Any]], limit: int | None, seed: int
) -> list[dict[str, Any]]:
    """Deterministic, task-stratified subsample; preserves original order."""
    if not limit or limit >= len(rows):
        return rows
    frame = pd.DataFrame(
        {
            "idx": range(len(rows)),
            "task_type": [row.get("task_type", "unknown") for row in rows],
        }
    )
    # Proportional per-task sampling without groupby.apply, which keeps behaviour
    # stable across the pandas versions in the training and analysis envs.
    total = len(frame)
    picks = []
    for _, group in frame.groupby("task_type"):
        count = min(len(group), max(1, round(limit * len(group) / total)))
        picks.append(group.sample(count, random_state=seed))
    picked = pd.concat(picks).sort_values("idx")
    return [rows[index] for index in picked["idx"].tolist()]


def _discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        match = _STEP_RE.search(path.name)
        if path.is_dir() and match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def _eval_one(
    label: str,
    step: int,
    kind: str,
    model_path: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    tokenizer_path: str | None,
) -> list[dict[str, Any]]:
    attempts = evaluate_model(
        f"{label}@{step}",
        str(model_path),
        rows,
        config,
        eos_token_ids=EVAL_EOS,
        tokenizer_path=tokenizer_path,
    )
    items = validation_items(attempts)
    out = [
        {
            "run_label": label,
            "kind": kind,
            "step": step,
            "scope": "overall",
            "task_type": "ALL",
            "avg_accuracy": float(items["avg_accuracy"].mean()),
            "valid_json_rate": float(attempts["valid_json"].mean()),
            "n_items": int(items["uid"].nunique()),
        }
    ]
    per_task = items.groupby("task_type", as_index=False)["avg_accuracy"].mean()
    json_task = attempts.groupby("task_type", as_index=False)["valid_json"].mean()
    per_task = per_task.merge(json_task, on="task_type", how="left")
    for _, row in per_task.iterrows():
        out.append(
            {
                "run_label": label,
                "kind": kind,
                "step": step,
                "scope": "task",
                "task_type": row["task_type"],
                "avg_accuracy": float(row["avg_accuracy"]),
                "valid_json_rate": float(row["valid_json"]),
                "n_items": int((items["task_type"] == row["task_type"]).sum()),
            }
        )
    return out


def run(
    run_specs: list[str],
    baseline_spec: str | None,
    config_path: str,
    limit: int | None,
    tokenizer_path: str | None,
    output: str,
) -> None:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    rows = _subsample(_load_jsonl(config["input_file"]), limit, seed)
    if tokenizer_path is None and baseline_spec is not None:
        tokenizer_path = str(_parse_spec(baseline_spec)[1])

    records: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] | None = None
    if baseline_spec is not None:
        label, path = _parse_spec(baseline_spec)
        baseline_rows = _eval_one(
            label, 0, "baseline", path, rows, config, tokenizer_path
        )
        records.extend(baseline_rows)

    for spec in run_specs:
        label, run_dir = _parse_spec(spec)
        checkpoints = _discover_checkpoints(run_dir)
        if not checkpoints:
            print(f"WARNING: no checkpoint-* dirs under {run_dir}")
        # Anchor the curve at the SFT init (step 0) so the rise is visible.
        if baseline_rows is not None:
            for record in baseline_rows:
                records.append({**record, "run_label": label, "kind": "grpo"})
        for step, checkpoint in checkpoints:
            records.extend(
                _eval_one(
                    label, step, "grpo", checkpoint, rows, config, tokenizer_path
                )
            )

    frame = pd.DataFrame.from_records(records)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Wrote {destination} ({len(frame)} rows, {len(rows)} eval items)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        default=[],
        help='GRPO run as "label=path" (repeatable), e.g. verifier=checkpoints/grpo_verifier',
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help='SFT init as "label=path", plotted as step 0, e.g. sft=checkpoints/eval/sft_multitask',
    )
    parser.add_argument("--config", default="configs/eval/validation.yaml")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Fallback tokenizer source for raw checkpoints (defaults to the baseline path).",
    )
    parser.add_argument("--output", default="results/validation/checkpoint_sweep.csv")
    args = parser.parse_args()
    if not args.runs:
        parser.error("at least one --run label=path is required")
    run(
        args.runs,
        args.baseline,
        args.config,
        args.limit,
        args.tokenizer_path,
        args.output,
    )


if __name__ == "__main__":
    main()
