from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

LADDER = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
]


def decide_escalation(
    validation_metrics: pd.DataFrame,
    current_model: str,
    margin_points: float = 2.0,
) -> dict[str, Any]:
    if "split" not in validation_metrics:
        raise ValueError("Escalation input must include split=validation provenance")
    if set(validation_metrics["split"]) != {"validation"}:
        raise ValueError("Escalation input must contain validation rows only")
    required = {"model_id", "avg_accuracy", "split"}
    if not required.issubset(validation_metrics.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(validation_metrics.columns))}")
    scores = validation_metrics.groupby("model_id")["avg_accuracy"].mean()
    delta_points = 100 * (
        float(scores["sft_multitask"]) - float(scores["instruct_qwen05"])
    )
    index = LADDER.index(current_model)
    escalate = delta_points < margin_points and index < len(LADDER) - 1
    return {
        "decision_source": "validation_only",
        "current_model": current_model,
        "next_model": LADDER[index + 1] if escalate else current_model,
        "sft_minus_instruct_points": delta_points,
        "required_margin_points": margin_points,
        "escalate": escalate,
    }


def _comparison_breakdown(path: Path, group_column: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    pivot = frame.pivot(
        index=group_column, columns="model_id", values="avg_accuracy"
    )
    if not {"instruct_qwen05", "sft_multitask"}.issubset(pivot.columns):
        return []
    pivot["delta_points"] = 100 * (
        pivot["sft_multitask"] - pivot["instruct_qwen05"]
    )
    return pivot.reset_index().to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_metrics", required=True)
    parser.add_argument("--current_model", choices=LADDER, required=True)
    parser.add_argument("--margin_points", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.validation_metrics)
    if "test" in path.name.lower() or "benchmark" in path.name.lower():
        raise ValueError("Refusing escalation decision from a test/benchmark-named file")
    decision = decide_escalation(
        pd.read_csv(path), args.current_model, args.margin_points
    )
    decision["task_type_breakdown"] = _comparison_breakdown(
        path.with_name("validation_task_type_breakdown.csv"), "task_type"
    )
    decision["multi_task_load_breakdown"] = _comparison_breakdown(
        path.with_name("validation_multitask_load_breakdown.csv"), "multi_task_load"
    )
    decision["complexity_breakdown"] = _comparison_breakdown(
        path.with_name("validation_complexity_breakdown.csv"), "complexity_bin"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
