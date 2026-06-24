from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    iterations: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for index in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[index] = statistic(sample)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def paired_delta(
    items: pd.DataFrame,
    left: str,
    right: str,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    pivot = items.pivot(index="uid", columns="model_id", values="avg_accuracy").dropna()
    differences = (pivot[right] - pivot[left]).to_numpy()
    low, high = bootstrap_ci(differences, iterations=iterations, seed=seed)
    left_mean, right_mean = pivot[left].mean(), pivot[right].mean()
    error_reduction = (
        (right_mean - left_mean) / (1 - left_mean) if left_mean < 1 else float("nan")
    )
    cohen_h = 2 * (
        math.asin(math.sqrt(right_mean)) - math.asin(math.sqrt(left_mean))
    )
    return {
        "estimate": differences.mean(),
        "ci_low": low,
        "ci_high": high,
        "relative_error_reduction": error_reduction,
        "cohen_h": cohen_h,
        "n_items": len(differences),
    }


def _item_level(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[frame["protocol"] == "A"]
        .groupby(["model_id", "uid"], as_index=False)
        .agg(avg_accuracy=("exact", "mean"))
    )


def run_tests(frame: pd.DataFrame, iterations: int = 10_000) -> pd.DataFrame:
    items = _item_level(frame)
    rows: list[dict] = []
    for model_id, group in items.groupby("model_id"):
        values = group["avg_accuracy"].to_numpy()
        low, high = bootstrap_ci(values, iterations=iterations)
        rows.append(
            {
                "comparison": f"{model_id}: overall",
                "estimate": values.mean(),
                "ci_low": low,
                "ci_high": high,
                "relative_error_reduction": np.nan,
                "cohen_h": np.nan,
                "n_items": len(values),
            }
        )
    comparisons = [
        ("sft_multitask", "grpo_verifier"),
        ("grpo_format", "grpo_verifier"),
    ]
    present = set(items["model_id"])
    for left, right in comparisons:
        if {left, right}.issubset(present):
            rows.append(
                {
                    "comparison": f"{right} - {left}",
                    **paired_delta(items, left, right, iterations),
                }
            )
    singles = {"sft_count", "sft_index", "sft_generation"}
    if singles | {"sft_multitask"} <= present:
        pivot = items.pivot(index="uid", columns="model_id", values="avg_accuracy")
        pivot["sft_single_task_avg"] = pivot[list(singles)].mean(axis=1)
        synthetic = (
            pivot[["sft_single_task_avg", "sft_multitask"]]
            .reset_index()
            .melt(id_vars="uid", var_name="model_id", value_name="avg_accuracy")
        )
        rows.append(
            {
                "comparison": "sft_multitask - sft_single_task_avg",
                **paired_delta(
                    synthetic,
                    "sft_single_task_avg",
                    "sft_multitask",
                    iterations,
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    files = sorted(Path(args.input_dir).glob("*_samples.parquet"))
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_tests(frame, args.iterations).to_csv(output, index=False)


if __name__ == "__main__":
    main()

