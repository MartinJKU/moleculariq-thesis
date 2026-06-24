from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)


def _save(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _controlled(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["protocol"] == "A"].copy()


def plot_all(tables_dir: Path, parsed_dir: Path, output_dir: Path) -> None:
    leaderboard = _controlled(pd.read_csv(tables_dir / "main_leaderboard.csv"))
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=leaderboard, x="model_id", y="avg_accuracy", ax=axis)
    axis.set(xlabel="", ylabel="avg_accuracy (Protocol A)")
    axis.tick_params(axis="x", rotation=35)
    _save(fig, output_dir, "barplot_main_leaderboard")

    task = _controlled(pd.read_csv(tables_dir / "task_type_breakdown.csv"))
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=task, x="task_type", y="avg_accuracy", hue="model_id", ax=axis)
    axis.set(xlabel="", ylabel="avg_accuracy")
    _save(fig, output_dir, "barplot_task_type_accuracy")

    load = _controlled(pd.read_csv(tables_dir / "multitask_load_breakdown.csv"))
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=load,
        x="multi_task_load",
        y="avg_accuracy",
        hue="model_id",
        marker="o",
        ax=axis,
    )
    _save(fig, output_dir, "lineplot_multitask_load")

    complexity = _controlled(pd.read_csv(tables_dir / "complexity_breakdown.csv"))
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=complexity,
        x="complexity_bin",
        y="avg_accuracy",
        hue="model_id",
        ax=axis,
    )
    _save(fig, output_dir, "barplot_complexity_bins")

    robustness = _controlled(pd.read_csv(tables_dir / "smiles_robustness_breakdown.csv"))
    robustness["representation"] = robustness.apply(
        lambda row: (
            "randomized + kekulized"
            if row["is_randomized"] and row["is_kekulized"]
            else "randomized"
            if row["is_randomized"]
            else "kekulized"
            if row["is_kekulized"]
            else "canonical"
        ),
        axis=1,
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=robustness,
        x="representation",
        y="avg_accuracy",
        hue="model_id",
        ax=axis,
    )
    _save(fig, output_dir, "barplot_smiles_robustness")

    fig, axis = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=leaderboard,
        x="valid_json_rate",
        y="exact_match_rate",
        hue="model_id",
        s=100,
        ax=axis,
    )
    _save(fig, output_dir, "scatter_format_vs_correctness")

    features = _controlled(pd.read_csv(tables_dir / "feature_breakdown.csv"))
    matrix = features.pivot_table(
        index="features", columns="model_id", values="avg_accuracy"
    )
    if "base_qwen05" in matrix:
        matrix = matrix.sort_values("base_qwen05")
    fig, axis = plt.subplots(figsize=(11, max(5, len(matrix) * 0.22)))
    sns.heatmap(matrix, cmap="viridis", vmin=0, vmax=1, ax=axis)
    _save(fig, output_dir, "heatmap_feature_accuracy")

    if {"sft_multitask", "grpo_verifier"} <= set(task["model_id"]):
        delta = task.pivot(
            index="task_type", columns="model_id", values="avg_accuracy"
        )
        delta["delta"] = delta["grpo_verifier"] - delta["sft_multitask"]
        fig, axis = plt.subplots(figsize=(7, 4))
        sns.barplot(x=delta.index, y=delta["delta"], ax=axis)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(ylabel="Delta avg_accuracy", xlabel="")
        _save(fig, output_dir, "barplot_delta_over_sft")

    log_files = sorted(Path("results/raw/grpo_logs").glob("*.jsonl"))
    if log_files:
        logs = []
        for path in log_files:
            frame = pd.read_json(path, lines=True)
            frame["run_id"] = path.stem
            logs.append(frame)
        log_frame = pd.concat(logs, ignore_index=True)
        for metric, stem in (
            ("mean_reward", "lineplot_grpo_reward"),
            ("valid_json_rate", "lineplot_grpo_valid_json"),
            ("exact_match_rate", "lineplot_grpo_exact_match"),
        ):
            if metric in log_frame:
                fig, axis = plt.subplots(figsize=(8, 4))
                sns.lineplot(
                    data=log_frame, x="step", y=metric, hue="run_id", ax=axis
                )
                _save(fig, output_dir, stem)

    efficiency_path = tables_dir / "efficiency_metrics.csv"
    if efficiency_path.exists():
        efficiency = _controlled(pd.read_csv(efficiency_path))
        metrics = [
            metric
            for metric in (
                "gpu_hours",
                "tokens_per_second",
                "samples_per_second",
                "benchmark_eval_time",
            )
            if metric in efficiency and efficiency[metric].notna().any()
        ]
        if metrics:
            fig, axes = plt.subplots(
                2, 2, figsize=(12, 8), squeeze=False
            )
            for axis, metric in zip(axes.flat, metrics, strict=False):
                sns.barplot(data=efficiency, x="model_id", y=metric, ax=axis)
                axis.tick_params(axis="x", rotation=35)
                axis.set(xlabel="", title=metric.replace("_", " "))
            for axis in list(axes.flat)[len(metrics):]:
                axis.set_visible(False)
            _save(fig, output_dir, "barplot_efficiency")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_dir", required=True)
    parser.add_argument("--parsed_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    plot_all(Path(args.tables_dir), Path(args.parsed_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
