import pandas as pd

from miqthesis.evaluation.run_validation import (
    aggregate_validation,
    validation_breakdown,
)


def test_validation_metrics_are_item_level_and_labelled():
    attempts = pd.DataFrame(
        {
            "model_id": ["a", "a", "a", "a"],
            "uid": ["x", "x", "y", "y"],
            "task_type": ["count"] * 4,
            "features": ["ring"] * 4,
            "complexity_bin": ["low"] * 4,
            "multi_task_load": [1] * 4,
            "exact": [0, 1, 1, 1],
        }
    )
    overall, breakdown = aggregate_validation(attempts)
    assert overall.iloc[0]["split"] == "validation"
    assert overall.iloc[0]["avg_accuracy"] == 0.75
    assert breakdown.iloc[0]["task_type"] == "count"
    assert validation_breakdown(attempts, "multi_task_load").iloc[0]["n_items"] == 2
