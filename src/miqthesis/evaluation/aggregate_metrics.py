from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _item_level(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_id",
        "protocol",
        "uid",
        "task_type",
        "features",
        "complexity_bin",
        "multi_task_load",
        "is_randomized",
        "is_kekulized",
    ]
    return (
        frame.groupby(keys, dropna=False, as_index=False)
        .agg(
            avg_accuracy=("exact", "mean"),
            exact_match_rate=("exact", "mean"),
            parse_rate=("prediction_answer_block", lambda values: values.notna().mean()),
            valid_json_rate=("valid_json", "mean"),
            partial_json_field_accuracy=("field_accuracy", "mean"),
            valid_smiles_rate=("valid_smiles", "mean"),
            constraint_satisfaction_rate=("constraint_satisfaction", "mean"),
            pass_at_1=("pass_at_1_component", "first"),
            pass_at_3=("pass_at_3_component", "first"),
            latency_per_sample=("latency", "mean"),
            num_prompt_tokens=("num_prompt_tokens", "mean"),
            num_completion_tokens=("num_completion_tokens", "mean"),
        )
    )


METRICS = [
    "avg_accuracy",
    "pass_at_1",
    "pass_at_3",
    "parse_rate",
    "valid_json_rate",
    "exact_match_rate",
    "partial_json_field_accuracy",
    "valid_smiles_rate",
    "constraint_satisfaction_rate",
    "latency_per_sample",
    "num_prompt_tokens",
    "num_completion_tokens",
]


def aggregate(frame: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
    groups = ["model_id", "protocol", *groups]
    return frame.groupby(groups, dropna=False, as_index=False)[METRICS].mean()


def aggregate_all(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    items = _item_level(frame)
    efficiency = aggregate(items, [])[
        [
            "model_id",
            "protocol",
            "latency_per_sample",
            "num_prompt_tokens",
            "num_completion_tokens",
        ]
    ].copy()
    total_time = (
        frame.groupby(["model_id", "protocol"], dropna=False)["latency"]
        .sum()
        .rename("benchmark_eval_time")
        .reset_index()
    )
    efficiency = efficiency.merge(total_time, on=["model_id", "protocol"], how="left")
    resource_rows = []
    for path in sorted(Path("results/raw/training_logs").glob("*.jsonl")):
        log = pd.read_json(path, lines=True)
        if log.empty:
            continue
        last = log.sort_values("wallclock_seconds").iloc[-1]
        resource_rows.append(
            {
                "model_id": path.stem,
                "gpu_hours": float(last.get("wallclock_seconds", 0))
                * float(last.get("gpu_count", 1))
                / 3600,
                "tokens_per_second": last.get("tokens_per_second"),
                "samples_per_second": last.get("samples_per_second"),
            }
        )
    if resource_rows:
        efficiency = efficiency.merge(
            pd.DataFrame(resource_rows), on="model_id", how="left"
        )
    else:
        efficiency["gpu_hours"] = pd.NA
        efficiency["tokens_per_second"] = pd.NA
        efficiency["samples_per_second"] = pd.NA
    tables = {
        "main_leaderboard.csv": aggregate(items, []),
        "task_type_breakdown.csv": aggregate(items, ["task_type"]),
        "feature_breakdown.csv": aggregate(items, ["features"]),
        "complexity_breakdown.csv": aggregate(items, ["complexity_bin"]),
        "multitask_load_breakdown.csv": aggregate(items, ["multi_task_load"]),
        "smiles_robustness_breakdown.csv": aggregate(
            items, ["is_randomized", "is_kekulized"]
        ),
        "task_complexity_breakdown.csv": aggregate(
            items, ["task_type", "complexity_bin"]
        ),
        "task_multitask_load_breakdown.csv": aggregate(
            items, ["task_type", "multi_task_load"]
        ),
        "efficiency_metrics.csv": efficiency,
    }
    return tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    files = sorted(Path(args.input_dir).glob("*_samples.parquet"))
    if not files:
        raise FileNotFoundError(f"No parsed sample parquet files in {args.input_dir}")
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in aggregate_all(frame).items():
        table.to_csv(output_dir / name, index=False)
        print(f"Wrote {output_dir / name}")
    log_files = sorted(Path("results/raw/grpo_logs").glob("*.jsonl"))
    if log_files:
        logs = []
        for path in log_files:
            log = pd.read_json(path, lines=True)
            log["run_id"] = path.stem
            logs.append(log)
        pd.concat(logs, ignore_index=True).to_csv(
            output_dir / "reward_training_summary.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=[
                "run_id",
                "step",
                "mean_reward",
                "std_reward",
                "valid_json_rate",
                "exact_match_rate",
                "field_accuracy",
                "valid_smiles_rate",
                "constraint_satisfaction_rate",
                "completion_length_mean",
            ]
        ).to_csv(output_dir / "reward_training_summary.csv", index=False)


if __name__ == "__main__":
    main()
