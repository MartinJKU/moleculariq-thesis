import json

from miqthesis.evaluation.parse_lm_eval_outputs import (
    _flatten_responses,
    find_official_metrics,
)


def test_response_flattening():
    assert _flatten_responses([[{"text": "a"}], ["b"]]) == ["a", "b"]


def test_reads_avg_accuracy_from_lm_eval_output(tmp_path):
    payload = {
        "results": {
            "moleculariq_pass_at_k": {
                "avg_accuracy,all": 0.5,
                "pass_at_1,all": 0.4,
            }
        }
    }
    path = tmp_path / "results_1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    metrics = find_official_metrics(tmp_path)
    assert metrics["avg_accuracy"] == 0.5
    assert metrics["pass_at_1"] == 0.4

