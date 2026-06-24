import pandas as pd

from miqthesis.evaluation.aggregate_metrics import aggregate_all


def _row(model, uid, repeat, exact, task="count", load=1):
    return {
        "model_id": model,
        "protocol": "A",
        "uid": uid,
        "repeat_index": repeat,
        "task_type": task,
        "features": "ring_count",
        "complexity_bin": "low",
        "multi_task_load": load,
        "is_randomized": False,
        "is_kekulized": False,
        "prediction_answer_block": "{}",
        "exact": exact,
        "field_accuracy": exact,
        "valid_json": 1,
        "valid_smiles": 0,
        "constraint_satisfaction": 0,
        "pass_at_1_component": exact if repeat == 0 else 0,
        "pass_at_3_component": 1,
        "latency": 1.0,
        "num_prompt_tokens": 10,
        "num_completion_tokens": 2,
    }


def test_repeat_aware_aggregation():
    frame = pd.DataFrame(
        [
            _row("m", "a", 0, 0),
            _row("m", "a", 1, 1),
            _row("m", "a", 2, 0),
            _row("m", "b", 0, 1),
            _row("m", "b", 1, 1),
            _row("m", "b", 2, 1),
        ]
    )
    leaderboard = aggregate_all(frame)["main_leaderboard.csv"].iloc[0]
    assert leaderboard["avg_accuracy"] == 4 / 6
    assert leaderboard["pass_at_1"] == 0.5
    assert leaderboard["pass_at_3"] == 1.0

