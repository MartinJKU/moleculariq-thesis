"""Build a qualitative before/after panel: for one representative item per task
family, show the prompt, the gold target, and each model's prediction side by
side. Reads the per-model validation attempts (results/validation/*_attempts.parquet)
and joins the prompt/target back from the validation JSONL by uid.

Picks, per task family, the item with the widest correctness spread across
models (some right, some wrong) so the contrast is informative.

Example
-------
python -m miqthesis.analysis.qualitative_panel \
  --attempts_dir results/validation \
  --val_jsonl data/processed/sft_multitask_val.jsonl \
  --output_dir results/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

MODEL_ORDER = [
    "instruct_qwen05",
    "sft_multitask",
    "grpo_format",
    "grpo_verifier",
]


def _load_attempts(attempts_dir: Path) -> pd.DataFrame:
    files = sorted(attempts_dir.glob("*_attempts.parquet"))
    if not files:
        raise FileNotFoundError(f"No *_attempts.parquet under {attempts_dir}")
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    # First repeat per (model, uid) is the displayed sample; exact is averaged.
    first = (
        frame.sort_values("repeat_index")
        .groupby(["model_id", "uid"], as_index=False)
        .first()[["model_id", "uid", "task_type", "prediction_raw"]]
    )
    accuracy = (
        frame.groupby(["model_id", "uid"], as_index=False)["exact"].mean()
    )
    return first.merge(accuracy, on=["model_id", "uid"])


def _prompt_target(val_jsonl: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    with val_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            user = [m for m in row.get("messages", []) if m.get("role") == "user"]
            lookup[row["uid"]] = {
                "question": user[-1]["content"] if user else "",
                "target": row.get("target"),
            }
    return lookup


def _ordered_models(models: list[str]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(m for m in models if m not in MODEL_ORDER)


def _truncate(text: Any, width: int = 220) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def build_panel(
    attempts_dir: Path, val_jsonl: Path, output_dir: Path, width: int
) -> None:
    attempts = _load_attempts(attempts_dir)
    lookup = _prompt_target(val_jsonl)
    models = _ordered_models(sorted(attempts["model_id"].unique()))
    output_dir.mkdir(parents=True, exist_ok=True)

    chosen: list[str] = []
    for task_type, group in attempts.groupby("task_type"):
        spread = group.groupby("uid")["exact"].agg(["max", "min", "count"])
        spread = spread[spread["count"] >= max(2, len(models) - 1)]
        if spread.empty:
            continue
        spread["range"] = spread["max"] - spread["min"]
        uid = spread.sort_values(["range", "max"], ascending=False).index[0]
        chosen.append(uid)

    lines = ["# Qualitative panel\n"]
    for uid in chosen:
        rows = attempts[attempts["uid"] == uid]
        meta = lookup.get(uid, {})
        task_type = rows["task_type"].iloc[0]
        lines.append(f"## {task_type}  (`{uid}`)\n")
        lines.append(f"**Prompt:** {_truncate(meta.get('question', ''), 400)}\n")
        lines.append(f"**Target:** `{_truncate(meta.get('target', ''), 200)}`\n")
        lines.append("| model | ✓ | prediction |")
        lines.append("|---|---|---|")
        for model in models:
            match = rows[rows["model_id"] == model]
            if match.empty:
                continue
            mark = "✅" if float(match["exact"].iloc[0]) >= 0.999 else "❌"
            prediction = _truncate(match["prediction_raw"].iloc[0], width)
            lines.append(f"| `{model}` | {mark} | {prediction} |")
        lines.append("")

    markdown = output_dir / "qualitative_panel.md"
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {markdown} ({len(chosen)} items, {len(models)} models)")

    _render_figure(attempts, lookup, chosen, models, output_dir, width)


def _render_figure(
    attempts: pd.DataFrame,
    lookup: dict[str, dict[str, Any]],
    chosen: list[str],
    models: list[str],
    output_dir: Path,
    width: int,
) -> None:
    import matplotlib.pyplot as plt

    if not chosen:
        return
    fig, axes = plt.subplots(
        len(chosen), 1, figsize=(11, 2.2 + 1.0 * len(models) * len(chosen) / 3),
        squeeze=False,
    )
    for axis, uid in zip(axes[:, 0], chosen):
        rows = attempts[attempts["uid"] == uid]
        meta = lookup.get(uid, {})
        text = [
            f"[{rows['task_type'].iloc[0]}]  {_truncate(meta.get('question', ''), 110)}",
            f"target: {_truncate(meta.get('target', ''), 90)}",
        ]
        for model in models:
            match = rows[rows["model_id"] == model]
            if match.empty:
                continue
            mark = "OK " if float(match["exact"].iloc[0]) >= 0.999 else "X  "
            text.append(
                f"{mark}{model:<16} {_truncate(match['prediction_raw'].iloc[0], 90)}"
            )
        axis.axis("off")
        axis.text(
            0.0,
            1.0,
            "\n".join(text),
            family="monospace",
            fontsize=8,
            va="top",
            ha="left",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "qualitative_panel.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "qualitative_panel.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts_dir", default="results/validation")
    parser.add_argument("--val_jsonl", default="data/processed/sft_multitask_val.jsonl")
    parser.add_argument("--output_dir", default="results/plots")
    parser.add_argument("--width", type=int, default=220)
    args = parser.parse_args()
    build_panel(
        Path(args.attempts_dir),
        Path(args.val_jsonl),
        Path(args.output_dir),
        args.width,
    )


if __name__ == "__main__":
    main()
