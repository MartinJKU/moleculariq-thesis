"""Poster figures that are not part of the standard make_plots pipeline:

* checkpoint sweep learning curve  (avg_accuracy vs GRPO step)         -> #1
* main leaderboard bars with bootstrap 95% CIs                          -> #2
* reward-component decomposition over training                         -> #6

Each figure is produced only if its input exists, so the same command works
whether or not the GRPO sweep / statistical tests have been run yet.

Example
-------
python -m miqthesis.analysis.poster_plots \
  --sweep_csv results/validation/checkpoint_sweep.csv \
  --stats_csv results/tables/statistical_tests.csv \
  --grpo_logs_dir results/raw/grpo_logs \
  --output_dir results/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from miqthesis.analysis.make_plots import _save  # consistent pdf+png saving

LEADERBOARD_ORDER = [
    "base_qwen05",
    "instruct_qwen05",
    "sft_multitask",
    "grpo_format",
    "grpo_verifier",
]


def plot_checkpoint_sweep(sweep_csv: Path, output_dir: Path) -> None:
    frame = pd.read_csv(sweep_csv)
    overall = frame[frame["scope"] == "overall"].copy()
    grpo = overall[overall["kind"] == "grpo"].sort_values(["run_label", "step"])
    baseline = overall[overall["kind"] == "baseline"]
    if grpo.empty:
        return

    fig, axis = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=grpo, x="step", y="avg_accuracy", hue="run_label", marker="o", ax=axis
    )
    if not baseline.empty:
        value = float(baseline["avg_accuracy"].mean())
        label = str(baseline["run_label"].iloc[0])
        axis.axhline(
            value, ls="--", color="0.4", lw=1.2, label=f"{label} (SFT) baseline"
        )

    max_step = int(grpo["step"].max())
    annotation = f"run truncated at step {max_step}"
    last_run = grpo[grpo["step"] == max_step]["run_label"].iloc[0]
    series = grpo[grpo["run_label"] == last_run].sort_values("step")["avg_accuracy"]
    if len(series) >= 2 and series.iloc[-1] > series.iloc[-2]:
        annotation += " — still climbing"
    axis.annotate(
        annotation,
        xy=(max_step, float(grpo["avg_accuracy"].max())),
        xytext=(0.45, 0.12),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="0.3"),
        fontsize=9,
    )
    axis.set(xlabel="GRPO step", ylabel="avg_accuracy (validation)")
    axis.legend(title="")
    _save(fig, output_dir, "lineplot_checkpoint_sweep")

    task = frame[frame["scope"] == "task"].copy()
    task = task[task["kind"] == "grpo"]
    if not task.empty:
        grid = sns.relplot(
            data=task.sort_values("step"),
            x="step",
            y="avg_accuracy",
            hue="run_label",
            col="task_type",
            kind="line",
            marker="o",
            height=3.5,
            facet_kws={"sharey": True},
        )
        grid.set_axis_labels("GRPO step", "avg_accuracy")
        grid.savefig(output_dir / "lineplot_checkpoint_sweep_by_task.png", dpi=300)
        grid.savefig(output_dir / "lineplot_checkpoint_sweep_by_task.pdf")
        plt.close(grid.fig)


def plot_leaderboard_ci(stats_csv: Path, output_dir: Path) -> None:
    frame = pd.read_csv(stats_csv)
    overall = frame[frame["comparison"].str.endswith(": overall")].copy()
    if overall.empty:
        return
    overall["model_id"] = overall["comparison"].str.replace(
        ": overall", "", regex=False
    )
    overall["order"] = overall["model_id"].apply(
        lambda model: LEADERBOARD_ORDER.index(model)
        if model in LEADERBOARD_ORDER
        else len(LEADERBOARD_ORDER)
    )
    overall = overall.sort_values(["order", "model_id"])
    lower = (overall["estimate"] - overall["ci_low"]).clip(lower=0)
    upper = (overall["ci_high"] - overall["estimate"]).clip(lower=0)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        overall["model_id"],
        overall["estimate"],
        yerr=[lower, upper],
        capsize=4,
        color=sns.color_palette("deep")[0],
    )
    axis.set(xlabel="", ylabel="avg_accuracy (Protocol A, 95% CI)")
    axis.tick_params(axis="x", rotation=30)
    _save(fig, output_dir, "barplot_leaderboard_ci")


def plot_reward_decomposition(grpo_logs_dir: Path, output_dir: Path) -> None:
    files = sorted(Path(grpo_logs_dir).glob("*.jsonl"))
    frames: dict[str, pd.DataFrame] = {}
    for path in files:
        frame = pd.read_json(path, lines=True)
        if frame.empty or "step" not in frame:
            continue
        # The reward fn logs once per generation batch, so many high-variance rows
        # share a step. Average per step and lightly smooth so trends are legible
        # instead of a solid noise band.
        agg = (
            frame.dropna(subset=["step"])
            .groupby("step", as_index=False)
            .mean(numeric_only=True)
            .sort_values("step")
        )
        window = max(1, len(agg) // 40)
        for column in agg.columns:
            if column != "step":
                agg[column] = agg[column].rolling(
                    window, min_periods=1, center=True
                ).mean()
        frames[path.stem] = agg
    if not frames:
        return

    # (a) Honest view: measured component rates over training, overlaid per run.
    components = [
        ("valid_json_rate", "valid JSON"),
        ("field_accuracy", "field accuracy"),
        ("exact_match_rate", "exact match"),
    ]
    optional = [
        ("valid_smiles_rate", "valid SMILES"),
        ("constraint_satisfaction_rate", "constraint frac"),
    ]
    fig, axes = plt.subplots(
        1, len(frames), figsize=(6 * len(frames), 4.5), squeeze=False
    )
    for axis, (run, frame) in zip(axes[0], frames.items()):
        reward_col = (
            "verifier_reward_mean" if "verifier" in run else "format_reward_mean"
        )
        if reward_col in frame:
            axis.plot(
                frame["step"],
                frame[reward_col],
                color="black",
                lw=2.2,
                label="reward (optimized)",
            )
        for column, label in components:
            if column in frame:
                axis.plot(frame["step"], frame[column], lw=1.6, label=label)
        for column, label in optional:
            if column in frame and frame[column].abs().sum() > 0:
                axis.plot(frame["step"], frame[column], lw=1.2, ls=":", label=label)
        axis.set(title=run, xlabel="step", ylim=(0, 1.02))
        axis.legend(fontsize=8)
    axes[0][0].set_ylabel("rate / reward")
    _save(fig, output_dir, "lineplot_reward_components")

    # (b) Intuition view: reconstructed weighted reward contributions at the
    # final logged step. NOTE has_tags is approximated by valid_json_rate (a
    # proxy), and rates are pooled across task families, so this is a mechanism
    # cartoon, not an exact accounting of the reward.
    weights = {
        "tags (approx)": ("valid_json_rate", 0.10),
        "valid_json": ("valid_json_rate", 0.15),
        "field_accuracy": ("field_accuracy", 0.25),
        "exact / success": ("exact_match_rate", 0.50),
    }
    rows = []
    for run, frame in frames.items():
        last = frame.iloc[-1]
        row = {"run": run}
        row.update(
            {
                name: weight * float(last.get(column, 0.0))
                for name, (column, weight) in weights.items()
            }
        )
        rows.append(row)
    stacked = pd.DataFrame(rows).set_index("run")
    fig, axis = plt.subplots(figsize=(1.8 * len(stacked) + 3, 4.5))
    bottom = pd.Series(0.0, index=stacked.index)
    for name in weights:
        axis.bar(stacked.index, stacked[name], bottom=bottom, label=name)
        bottom = bottom + stacked[name]
    axis.set(ylabel="reconstructed reward mass", xlabel="")
    axis.legend(title="component", fontsize=8)
    _save(fig, output_dir, "barplot_reward_decomposition_final")


def plot_training_curves(grpo_logs_dir: Path, output_dir: Path) -> None:
    """GRPO training curves (#4) straight from the per-step logs, so they render
    with no GPU and no Protocol A data (the standard make_plots also draws these,
    but only once the leaderboard tables exist)."""
    files = sorted(Path(grpo_logs_dir).glob("*.jsonl"))
    frames = []
    for path in files:
        frame = pd.read_json(path, lines=True)
        if frame.empty:
            continue
        frame["run_id"] = path.stem
        frames.append(frame)
    if not frames:
        return
    # Average the per-batch reward logs to one row per (run, step) so the lines
    # are clean rather than a wide bootstrap band over hundreds of noisy points.
    logs = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["step"])
        .groupby(["run_id", "step"], as_index=False)
        .mean(numeric_only=True)
    )
    for metric, stem in (
        ("mean_reward", "lineplot_grpo_reward"),
        ("valid_json_rate", "lineplot_grpo_valid_json"),
        ("exact_match_rate", "lineplot_grpo_exact_match"),
    ):
        if metric in logs:
            fig, axis = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=logs, x="step", y=metric, hue="run_id", ax=axis)
            axis.set(xlabel="GRPO step", ylabel=metric.replace("_", " "))
            _save(fig, output_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_csv", default="results/validation/checkpoint_sweep.csv")
    parser.add_argument("--stats_csv", default="results/tables/statistical_tests.csv")
    parser.add_argument("--grpo_logs_dir", default="results/raw/grpo_logs")
    parser.add_argument("--output_dir", default="results/plots")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if Path(args.sweep_csv).exists():
        plot_checkpoint_sweep(Path(args.sweep_csv), output_dir)
    else:
        print(f"skip checkpoint sweep: {args.sweep_csv} not found")
    if Path(args.stats_csv).exists():
        plot_leaderboard_ci(Path(args.stats_csv), output_dir)
    else:
        print(f"skip leaderboard CI: {args.stats_csv} not found")
    if Path(args.grpo_logs_dir).exists():
        plot_training_curves(Path(args.grpo_logs_dir), output_dir)
        plot_reward_decomposition(Path(args.grpo_logs_dir), output_dir)
    else:
        print(f"skip training curves + decomposition: {args.grpo_logs_dir} not found")


if __name__ == "__main__":
    main()
