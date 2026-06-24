from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
from pathlib import Path
from typing import Any

from miqthesis.constants import QWEN_CHAT_TEMPLATE
from miqthesis.training.callbacks import ResourceLoggingCallback
from miqthesis.training.rewards import (
    _completion_text,
    _generation_details,
    compare_json,
    format_reward,
    parse_json_answer,
    verifier_reward_value,
)
from miqthesis.training.utils import (
    assert_full_parameter_config,
    load_yaml,
    require_local_model,
    set_seed,
    write_json,
)


def make_logged_reward(reward_type: str, run_id: str):
    output_path = Path("results/raw/grpo_logs") / f"{run_id}.jsonl"

    def reward_function(
        completions: list[Any],
        target: list[Any],
        task_type: list[str],
        constraints: list[Any] | None = None,
        trainer_state: Any = None,
        **_: Any,
    ) -> list[float]:
        constraints_list = constraints or [None] * len(completions)
        texts = [_completion_text(completion) for completion in completions]
        format_values = [format_reward(text) for text in texts]
        verifier_values = [
            verifier_reward_value(text, tgt, task, constraint)
            for text, tgt, task, constraint in zip(
                texts, target, task_type, constraints_list, strict=True
            )
        ]
        valid_json: list[float] = []
        exact: list[float] = []
        field_accuracy: list[float] = []
        valid_smiles: list[float] = []
        constraint_satisfaction: list[float] = []
        for text, tgt, task, constraint in zip(
            texts, target, task_type, constraints_list, strict=True
        ):
            parsed = parse_json_answer(text)
            valid_json.append(float(parsed is not None))
            if "generation" in task or "constraint" in task:
                valid, fraction, success = _generation_details(parsed, constraint)
                valid_smiles.append(valid)
                constraint_satisfaction.append(fraction)
                exact.append(success)
                field_accuracy.append(fraction)
            else:
                comparison = compare_json(parsed, tgt)
                valid_smiles.append(0.0)
                constraint_satisfaction.append(0.0)
                exact.append(float(comparison["exact"]))
                field_accuracy.append(float(comparison["field_accuracy"]))
        selected = format_values if reward_type == "format" else verifier_values
        if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0":
            payload = {
                "step": getattr(trainer_state, "global_step", None),
                "mean_reward": statistics.mean(selected),
                "std_reward": statistics.pstdev(selected),
                "format_reward_mean": statistics.mean(format_values),
                "verifier_reward_mean": statistics.mean(verifier_values),
                "valid_json_rate": statistics.mean(valid_json),
                "exact_match_rate": statistics.mean(exact),
                "field_accuracy": statistics.mean(field_accuracy),
                "valid_smiles_rate": statistics.mean(valid_smiles),
                "constraint_satisfaction_rate": statistics.mean(
                    constraint_satisfaction
                ),
                "completion_length_mean": statistics.mean(map(len, texts)),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return selected

    reward_function.__name__ = f"{reward_type}_reward"
    return reward_function


def train(
    config_path: str,
    model_name_or_path: str | None = None,
    output_dir: str | None = None,
    run_id: str | None = None,
) -> None:
    config = load_yaml(config_path)
    if model_name_or_path:
        config["model_name_or_path"] = model_name_or_path
    if output_dir:
        config["output_dir"] = output_dir
    if run_id:
        config["run_id"] = run_id
    assert_full_parameter_config(config)
    config["model_name_or_path"] = str(
        require_local_model(config["model_name_or_path"])
    )
    set_seed(int(config.get("seed", 42)))
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], use_fast=True, local_files_only=True
    )
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"torch_dtype": "auto"}
    if config.get("bf16"):
        import torch

        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"], local_files_only=True, **model_kwargs
    )
    model.config.use_cache = False
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Full-parameter invariant violated: some parameters are frozen")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    write_json(
        f"{config['output_dir']}/run_manifest.json",
        {
            "config": config,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
            "full_parameter_finetuning": trainable_parameters == total_parameters,
        },
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": config["train_file"],
            "validation": config["validation_file"],
        },
    )
    reward = make_logged_reward(config["reward_type"], config["run_id"])
    grpo_kwargs = dict(
        output_dir=config["output_dir"],
        run_name=config["run_id"],
        max_completion_length=int(config["max_completion_length"]),
        num_generations=int(config["num_generations"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_train_epochs"]),
        beta=float(config["beta"]),
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        bf16=bool(config["bf16"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        logging_steps=int(config["logging_steps"]),
        eval_steps=int(config["eval_steps"]),
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        report_to=config.get("report_to", "none"),
        seed=int(config.get("seed", 42)),
    )
    evaluation_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(GRPOConfig).parameters
        else "evaluation_strategy"
    )
    grpo_kwargs[evaluation_key] = "steps"
    if "max_prompt_length" in inspect.signature(GRPOConfig).parameters:
        grpo_kwargs["max_prompt_length"] = int(config["max_prompt_length"])
    if "include_num_input_tokens_seen" in inspect.signature(GRPOConfig).parameters:
        grpo_kwargs["include_num_input_tokens_seen"] = True
    arguments = GRPOConfig(**grpo_kwargs)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[
            ResourceLoggingCallback(
                f"results/raw/training_logs/{config['run_id']}.jsonl"
            )
        ],
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--run_id")
    args = parser.parse_args()
    train(args.config, args.model_name_or_path, args.output_dir, args.run_id)


if __name__ == "__main__":
    main()
