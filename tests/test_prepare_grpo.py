import jsonlines

from miqthesis.data.prepare_grpo import _write_systematic_sample


def test_grpo_sampling_writes_exact_size_without_materializing_source(tmp_path):
    source = tmp_path / "sft_multitask_train.jsonl"
    with jsonlines.open(source, mode="w") as writer:
        writer.write_all(
            {
                "uid": str(index),
                "target": '{"ring_count": 1}',
                "task_type": "count",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": f"question {index}"},
                    {"role": "assistant", "content": '<answer>{"ring_count": 1}</answer>'},
                ],
            }
            for index in range(10)
        )
    destination = tmp_path / "grpo_multitask_train.jsonl"
    assert _write_systematic_sample(destination, source, 4) == 4
    with jsonlines.open(destination) as reader:
        rows = list(reader)
    assert len(rows) == 4
    assert all("prompt" in row and "messages" not in row for row in rows)
