import json

import pandas as pd
import pytest

from miqthesis.analysis.model_selection import decide_escalation
from miqthesis.evaluation.run_lm_eval import (
    assert_eval_stop_tokens,
    build_command,
    controlled_task_name,
    write_task_override,
)


def _write_generation_config(directory, eos):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "generation_config.json").write_text(
        json.dumps({"eos_token_id": eos}), encoding="utf-8"
    )
    return directory


def test_eval_rejects_chat_checkpoint_without_im_end_eos(tmp_path):
    checkpoint = _write_generation_config(tmp_path / "raw", [151643])
    with pytest.raises(ValueError, match="151645"):
        assert_eval_stop_tokens("grpo_verifier_step1000", {"chat_template": True}, str(checkpoint))


def test_eval_accepts_controlled_eos_and_skips_non_chat_models(tmp_path):
    prepared = _write_generation_config(tmp_path / "prepared", [151645, 151643])
    # Controlled checkpoint passes; a non-chat baseline is skipped even if its path
    # has no generation_config at all.
    assert_eval_stop_tokens("grpo_verifier", {"chat_template": True}, str(prepared))
    assert_eval_stop_tokens("base_qwen05", {"chat_template": False}, str(tmp_path / "missing"))


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
        "task: moleculariq_pass_at_k\n"
        "process_docs: !function task_processor.process_docs\n"
        "doc_to_target: target\n"
        "repeats: 3\n",
        encoding="utf-8",
    )
    override = write_task_override(
        tmp_path / "eval", "moleculariq_pass_at_k", 1, tmp_path
    )
    text = override.read_text(encoding="utf-8")
    assert "repeats: 1" in text
    assert f"task: {controlled_task_name('moleculariq_pass_at_k', 1)}" in text
    # The base task is inherited wholesale via include; the override must not
    # copy callable fields itself (the harness resolves them from the included
    # file when the task is loaded by name, see build_command).
    assert f'include: "{(task_dir / "moleculariq_pass_at_k.yaml").resolve().as_posix()}"' in text
    assert "process_docs" not in text
    assert "!function" not in text


def test_repeat_override_invoked_by_name_via_include_path(tmp_path):
    include_path = tmp_path / "task_override"
    command = build_command(
        "checkpoint",
        {"eval_task": "moleculariq_pass_at_k", "chat_template": True},
        {
            "protocol": "A",
            "task": "moleculariq_pass_at_k",
            "dtype": "bfloat16",
            "batch_size": "auto",
            "system_prompt": "fixed",
            "task_override_include_path": str(include_path),
            "task_override_name": "moleculariq_pass_at_k_controlled_repeats_3",
            "generation": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "max_new_tokens": 512,
            },
        },
        tmp_path,
    )
    # The override must be referenced by NAME through --include_path; passing it
    # as a .yaml file path makes the harness skip !function resolution.
    assert "--include_path" in command
    assert command[command.index("--include_path") + 1] == str(include_path)
    task_value = command[command.index("--tasks") + 1]
    assert task_value == "moleculariq_pass_at_k_controlled_repeats_3"
    assert not task_value.endswith(".yaml")


def test_no_override_passes_task_name_without_include_path(tmp_path):
    command = build_command(
        "checkpoint",
        {"eval_task": "moleculariq_pass_at_k", "chat_template": True},
        {
            "protocol": "A",
            "task": "moleculariq_pass_at_k",
            "batch_size": "auto",
            "generation": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "max_new_tokens": 512,
            },
        },
        tmp_path,
    )
    assert "--include_path" not in command
    assert command[command.index("--tasks") + 1] == "moleculariq_pass_at_k"


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
