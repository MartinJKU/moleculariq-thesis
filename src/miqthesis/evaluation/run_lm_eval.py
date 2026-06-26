from __future__ import annotations

import argparse
import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miqthesis.constants import IM_END_TOKEN_ID
from miqthesis.training.utils import load_yaml, require_cuda, require_local_model


def _provenance(eval_repo: str | None) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        version = subprocess.run(
            ["lm_eval", "--version"], capture_output=True, text=True, check=True
        )
        provenance["lm_eval_version"] = version.stdout.strip() or version.stderr.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        provenance["lm_eval_version"] = "unknown"
    if eval_repo and Path(eval_repo).exists():
        try:
            commit = subprocess.run(
                ["git", "-C", eval_repo, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            provenance["moleculariq_eval_commit"] = commit.stdout.strip()
        except subprocess.CalledProcessError:
            provenance["moleculariq_eval_commit"] = "unknown"
    return provenance


def build_command(
    model_path: str,
    model_config: dict[str, Any],
    eval_config: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    protocol = str(eval_config["protocol"]).upper()
    command = [
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        (
            f"pretrained={model_path}"
            f",dtype={eval_config.get('dtype', 'bfloat16')}"
            f",max_num_batched_tokens={eval_config.get('max_num_batched_tokens', 32768)}"
            f",max_num_seqs={eval_config.get('max_num_seqs', 256)}"
        ),
    ]
    # A repeat-controlled task must be referenced by NAME via --include_path, not
    # by file path. The harness loads a .yaml passed straight to --tasks with
    # resolve_func=False, so its !function fields (process_docs, ...) stay plain
    # strings and crash with "'str' object is not callable". Tasks discovered
    # through --include_path are instead built from their yaml_path with
    # resolve_func=True, which resolves those references to callables.
    override_name = eval_config.get("task_override_name")
    if override_name:
        command += ["--include_path", str(eval_config["task_override_include_path"])]
        task_value = override_name
    else:
        task_value = str(model_config.get("eval_task", eval_config["task"]))
    command += [
        "--tasks",
        task_value,
        "--batch_size",
        str(eval_config.get("batch_size", 1)),
        "--log_samples",
        "--output_path",
        str(output_dir),
    ]
    if protocol == "A":
        if model_config.get("chat_template"):
            command.append("--apply_chat_template")
            prompt = eval_config.get("system_prompt")
            if prompt:
                command.extend(["--system_instruction", str(prompt)])
        generation = eval_config["generation"]
        gen_kwargs = {
            "temperature": generation["temperature"],
            "top_p": generation.get("top_p"),
            "top_k": generation.get("top_k"),
            "max_gen_toks": generation["max_new_tokens"],
        }
        serialized = ",".join(
            f"{key}={value}" for key, value in gen_kwargs.items() if value is not None
        )
        command.extend(["--gen_kwargs", serialized])
    elif protocol == "B":
        if model_config.get("chat_template"):
            command.append("--apply_chat_template")
            prompt = eval_config.get("system_prompt")
            if prompt:
                command.extend(["--system_instruction", str(prompt)])
    else:
        raise ValueError(f"Unknown evaluation protocol: {protocol}")
    return command


def assert_eval_stop_tokens(
    model_id: str, model_config: dict[str, Any], model_path: str
) -> None:
    """Fail before launch if a chat-template checkpoint cannot stop the assistant turn.

    vLLM stops generation on the checkpoint's ``generation_config`` ``eos_token_id``.
    A raw training checkpoint inherits the base model's eos (``<|endoftext|>`` only),
    so the model never stops at ``<|im_end|>``, over-generates a whole fake
    conversation up to ``max_gen_toks``, and answer extraction collapses.
    ``prepare_checkpoint`` writes the controlled generation config that lists
    ``<|im_end|>``; this guard ensures only such checkpoints reach Protocol A/B and
    turns a silent score collapse into an actionable error.
    """
    if not model_config.get("chat_template"):
        return
    generation_path = Path(model_path) / "generation_config.json"
    if not generation_path.exists():
        raise ValueError(
            f"{model_id}: chat-template evaluation requires a generation_config.json "
            f"at {model_path}. Stage the checkpoint with "
            "`python -m miqthesis.training.prepare_checkpoint`."
        )
    eos = json.loads(generation_path.read_text(encoding="utf-8")).get("eos_token_id")
    eos_ids = eos if isinstance(eos, list) else [eos]
    if IM_END_TOKEN_ID not in eos_ids:
        raise ValueError(
            f"{model_id}: checkpoint at {model_path} has eos_token_id={eos!r}, which "
            f"is missing <|im_end|> ({IM_END_TOKEN_ID}). Under vLLM the model will not "
            "stop at the end of the assistant turn and will over-generate, breaking "
            "answer extraction. Evaluate a checkpoint staged by "
            "`python -m miqthesis.training.prepare_checkpoint` (which writes the "
            "controlled generation config), not a raw training checkpoint."
        )


def controlled_task_name(base_task: str, repeats: int) -> str:
    """Name of the repeat-controlled task variant inherited from ``base_task``."""
    return f"{base_task}_controlled_repeats_{repeats}"


def write_task_override(
    eval_repo: str | Path,
    base_task: str,
    repeats: int,
    output_dir: str | Path,
) -> Path:
    task_file = Path(eval_repo) / "lm_eval" / "tasks" / "moleculariq" / f"{base_task}.yaml"
    if not task_file.exists():
        raise FileNotFoundError(f"Cannot build repeat override; missing {task_file}")

    # Inherit the official task wholesale via include and override only repeats.
    # The override is referenced by task name through --include_path (see
    # build_command), so the harness builds it from this file with
    # resolve_func=True; the include then resolves the base task's !function
    # fields to callables and carries every other field (doc_to_target, metrics,
    # ...) through unchanged. Nothing else needs to be re-specified here.
    override = Path(output_dir) / f"{base_task}_repeats_{repeats}.yaml"
    override.write_text(
        "\n".join(
            [
                f'include: "{task_file.resolve().as_posix()}"',
                f"task: {controlled_task_name(base_task, repeats)}",
                f"repeats: {repeats}",
                "metadata:",
                "  version: controlled-1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return override


def run(
    config_path: str,
    models_path: str,
    model_ids: list[str] | None = None,
    dry_run: bool = False,
    eval_repo: str | None = None,
) -> None:
    if not dry_run:
        require_cuda("MolecularIQ evaluation")
    eval_config = load_yaml(config_path)
    models = load_yaml(models_path)["models"]
    selected = model_ids or [
        model_id for model_id, value in models.items() if not value.get("optional")
    ]
    if str(eval_config["protocol"]).upper() == "B" and len(selected) != 1:
        raise ValueError("Protocol B is best-model-only and requires exactly one model")
    for model_id in selected:
        model_config = models[model_id]
        output_dir = Path(eval_config["output_root"]) / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_eval_config = copy.deepcopy(eval_config)
        repeats = model_eval_config.get("generation", {}).get("num_repeats")
        if str(model_eval_config["protocol"]).upper() == "A" and repeats is not None:
            if not eval_repo:
                raise ValueError(
                    "Protocol A repeat control requires --eval_repo for a task override"
                )
            base_task = model_config.get("eval_task", model_eval_config["task"])
            override_dir = output_dir / "task_override"
            override_dir.mkdir(parents=True, exist_ok=True)
            write_task_override(eval_repo, base_task, int(repeats), override_dir)
            model_eval_config["task_override_include_path"] = str(override_dir)
            model_eval_config["task_override_name"] = controlled_task_name(
                base_task, int(repeats)
            )
        protocol = str(model_eval_config["protocol"]).upper()
        model_path = (
            model_config.get("leaderboard_model_path", model_config["model_path"])
            if protocol == "B"
            else model_config["model_path"]
        )
        model_path = str(require_local_model(model_path))
        assert_eval_stop_tokens(model_id, model_config, model_path)
        command = build_command(model_path, model_config, model_eval_config, output_dir)
        manifest = {
            "model_id": model_id,
            "model": {**model_config, "resolved_model_path": model_path},
            "evaluation": model_eval_config,
            "command": command,
            "provenance": _provenance(eval_repo),
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(" ".join(command))
        if not dry_run:
            subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--model_id", action="append", dest="model_ids")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--eval_repo")
    args = parser.parse_args()
    run(
        args.config,
        args.models,
        args.model_ids,
        args.dry_run,
        args.eval_repo,
    )


if __name__ == "__main__":
    main()
