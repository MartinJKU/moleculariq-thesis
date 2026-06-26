import pandas as pd
import pytest

from miqthesis.analysis.model_selection import decide_escalation
from miqthesis.evaluation.run_lm_eval import build_command, write_task_override


def test_protocol_a_pins_prompt_and_generation(tmp_path):
    command = build_command(
        "checkpoint",
        {"eval_task": "moleculariq_pass_at_k", "chat_template": True},
        {
            "protocol": "A",
            "task": "moleculariq_pass_at_k",
            "dtype": "bfloat16",
            "batch_size": "auto",
            "system_prompt": "fixed",
            "generation": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "max_new_tokens": 512,
            },
        },
        tmp_path,
    )
    assert "--system_instruction" in command
    assert "--gen_kwargs" in command


def test_protocol_b_does_not_override_native_generation(tmp_path):
    command = build_command(
        "checkpoint",
        {"eval_task": "moleculariq_pass_at_k", "chat_template": True},
        {
            "protocol": "B",
            "task": "moleculariq_pass_at_k",
            "dtype": "bfloat16",
            "batch_size": "auto",
            "system_prompt": "official",
        },
        tmp_path,
    )
    assert "--gen_kwargs" not in command
    assert "--system_instruction" in command


def test_repeat_override_inherits_official_task(tmp_path):
    task_dir = tmp_path / "eval" / "lm_eval" / "tasks" / "moleculariq"
    task_dir.mkdir(parents=True)
    (task_dir / "moleculariq_pass_at_k.yaml").write_text(
        "task: moleculariq_pass_at_k\nrepeats: 3\n", encoding="utf-8"
    )
    override = write_task_override(
        tmp_path / "eval", "moleculariq_pass_at_k", 1, tmp_path
    )
    text = override.read_text(encoding="utf-8")
    assert "repeats: 1" in text
    assert "include:" in text


def test_repeat_override_retags_function_fields(tmp_path):
    task_dir = tmp_path / "eval" / "lm_eval" / "tasks" / "moleculariq"
    task_dir.mkdir(parents=True)
    (task_dir / "moleculariq_pass_at_k.yaml").write_text(
        "task: moleculariq_pass_at_k\n"
        "process_docs: !function task_processor.process_docs\n"
        "doc_to_text: !function task_processor.doc_to_text\n"
        "doc_to_target: target\n"
        "process_results: !function task_processor.process_results_pass_at_k\n"
        "repeats: 3\n",
        encoding="utf-8",
    )
    override = write_task_override(
        tmp_path / "eval", "moleculariq_pass_at_k", 1, tmp_path
    )
    text = override.read_text(encoding="utf-8")
    abs_dir = task_dir.resolve().as_posix()
    # !function fields must keep the tag and be qualified with the base task's
    # absolute dir; without the tag lm_eval leaves them as strings and calling
    # them raises "'str' object is not callable".
    assert f"process_docs: !function {abs_dir}/task_processor.process_docs" in text
    assert f"doc_to_text: !function {abs_dir}/task_processor.doc_to_text" in text
    assert (
        f"process_results: !function {abs_dir}/task_processor.process_results_pass_at_k"
        in text
    )
    # Plain-string fields must not be re-tagged or path-prefixed; doc_to_target
    # (a column name) is inherited verbatim through the include instead.
    assert "doc_to_target: !function" not in text
    assert f"{abs_dir}/target" not in text


def test_escalation_is_validation_only():
    validation = pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "model_id": ["instruct_qwen05", "sft_multitask"],
            "avg_accuracy": [0.40, 0.41],
        }
    )
    decision = decide_escalation(validation, "Qwen/Qwen2.5-0.5B")
    assert decision["escalate"] is True
    invalid = validation.copy()
    invalid.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        decide_escalation(invalid, "Qwen/Qwen2.5-0.5B")
