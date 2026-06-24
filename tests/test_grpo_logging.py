import json

from miqthesis.training.train_grpo import make_logged_reward


def test_logged_reward_records_thesis_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reward = make_logged_reward("verifier", "test")
    values = reward(
        completions=['<answer>{"ring_count": 1}</answer>'],
        target=['{"ring_count": 1}'],
        task_type=["count"],
        constraints=[None],
    )
    assert values == [1.0]
    payload = json.loads(
        (tmp_path / "results/raw/grpo_logs/test.jsonl").read_text().strip()
    )
    assert payload["valid_json_rate"] == 1.0
    assert payload["exact_match_rate"] == 1.0
