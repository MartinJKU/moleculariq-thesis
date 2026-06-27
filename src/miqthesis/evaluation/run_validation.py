from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from miqthesis.constants import QWEN_CHAT_TEMPLATE
from miqthesis.evaluation.verify_outputs import verify_prediction
from miqthesis.training.utils import (
    load_yaml,
    require_cuda,
    require_local_model,
    transformers_dtype_kwargs,
)


def _batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validation_items(attempts: pd.DataFrame) -> pd.DataFrame:
    item_columns = [
        "model_id",
        "uid",
        "task_type",
        "features",
        "complexity_bin",
        "multi_task_load",
    ]
    return (
        attempts.groupby(item_columns, dropna=False, as_index=False)
        .agg(avg_accuracy=("exact", "mean"))
        .assign(split="validation")
    )


def aggregate_validation(attempts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = validation_items(attempts)
    overall = (
        items.groupby(["model_id", "split"], as_index=False)
        .agg(avg_accuracy=("avg_accuracy", "mean"), n_items=("uid", "nunique"))
    )
    breakdown = (
        items.groupby(["model_id", "split", "task_type"], as_index=False)
        .agg(avg_accuracy=("avg_accuracy", "mean"), n_items=("uid", "nunique"))
    )
    return overall, breakdown


def validation_breakdown(attempts: pd.DataFrame, column: str) -> pd.DataFrame:
    items = validation_items(attempts)
    return (
        items.groupby(["model_id", "split", column], dropna=False, as_index=False)
        .agg(avg_accuracy=("avg_accuracy", "mean"), n_items=("uid", "nunique"))
    )


def evaluate_model(
    model_id: str,
    model_path: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    eos_token_ids: list[int] | None = None,
    tokenizer_path: str | None = None,
) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_cuda("Validation")
    model_path = str(require_local_model(model_path))
    # Raw training checkpoints (checkpoint-XXXX) sometimes ship without tokenizer
    # files; fall back to an explicit tokenizer source (e.g. the SFT init) so a
    # checkpoint sweep does not require re-staging each intermediate step.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, local_files_only=True
        )
    except OSError:
        if tokenizer_path is None:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, local_files_only=True
        )
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if config.get("dtype") == "bfloat16" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        local_files_only=True,
        **transformers_dtype_kwargs(dtype),
    )
    model.eval()
    generation = config["generation"]
    repeats = int(generation["num_repeats"])
    records: list[dict[str, Any]] = []
    for batch in _batches(rows, int(config["batch_size"])):
        prompts = [
            tokenizer.apply_chat_template(
                row["messages"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generate_kwargs: dict[str, Any] = dict(
            do_sample=float(generation["temperature"]) > 0,
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            top_k=int(generation["top_k"]),
            max_new_tokens=int(generation["max_new_tokens"]),
            num_return_sequences=repeats,
            pad_token_id=tokenizer.pad_token_id,
        )
        # A raw checkpoint inherits the base generation_config whose only EOS is
        # <|endoftext|>; without <|im_end|> the assistant turn never stops under
        # greedy/sampling decoding. Force both so intermediate checkpoints halt.
        if eos_token_ids is not None:
            generate_kwargs["eos_token_id"] = eos_token_ids
        with torch.no_grad():
            outputs = model.generate(**encoded, **generate_kwargs)
        prompt_width = encoded["input_ids"].shape[1]
        completions = tokenizer.batch_decode(
            outputs[:, prompt_width:], skip_special_tokens=True
        )
        for completion_index, completion in enumerate(completions):
            row = batch[completion_index // repeats]
            verification = verify_prediction(
                completion,
                row["target"],
                row["task_type"],
                row.get("constraints"),
            )
            records.append(
                {
                    "model_id": model_id,
                    "split": "validation",
                    "uid": row["uid"],
                    "repeat_index": completion_index % repeats,
                    "task_type": row["task_type"],
                    "features": row.get("features", "unknown"),
                    "complexity_bin": row.get("complexity_bin", "unknown"),
                    "multi_task_load": row.get("multi_task_load", 1),
                    "prediction_raw": completion,
                    "exact": verification["exact"],
                    "field_accuracy": verification["field_accuracy"],
                    "valid_json": verification["valid_json"],
                }
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(records)


def _subsample_rows(
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
    total = len(frame)
    picks = []
    for _, group in frame.groupby("task_type"):
        count = min(len(group), max(1, round(limit * len(group) / total)))
        picks.append(group.sample(count, random_state=seed))
    picked = pd.concat(picks).sort_values("idx")
    return [rows[index] for index in picked["idx"].tolist()]


def run(
    config_path: str,
    models_path: str,
    model_ids: list[str],
    limit: int | None = None,
) -> None:
    config = load_yaml(config_path)
    models = load_yaml(models_path)["models"]
    rows = _subsample_rows(
        _load_jsonl(config["input_file"]), limit, int(config.get("seed", 42))
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for model_id in model_ids:
        model_path = models[model_id]["model_path"]
        frame = evaluate_model(model_id, model_path, rows, config)
        frame.to_parquet(output_dir / f"{model_id}_attempts.parquet", index=False)
        attempts.append(frame)
    combined = pd.concat(attempts, ignore_index=True)
    overall, breakdown = aggregate_validation(combined)
    overall.to_csv(output_dir / "validation_metrics.csv", index=False)
    breakdown.to_csv(output_dir / "validation_task_type_breakdown.csv", index=False)
    validation_breakdown(combined, "multi_task_load").to_csv(
        output_dir / "validation_multitask_load_breakdown.csv", index=False
    )
    validation_breakdown(combined, "complexity_bin").to_csv(
        output_dir / "validation_complexity_breakdown.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/validation.yaml")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument(
        "--model_id",
        action="append",
        dest="model_ids",
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most this many items (task-stratified). Default: all.",
    )
    args = parser.parse_args()
    run(
        args.config,
        args.models,
        args.model_ids or ["instruct_qwen05", "sft_multitask"],
        args.limit,
    )


if __name__ == "__main__":
    main()
