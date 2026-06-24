from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from typing import Any

from miqthesis.constants import QWEN_CHAT_TEMPLATE
from miqthesis.training.callbacks import ResourceLoggingCallback
from miqthesis.training.utils import (
    assert_full_parameter_config,
    load_yaml,
    set_seed,
    write_json,
)


@dataclass
class CausalChatCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [
            self.tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            for example in examples
        ]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


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
    set_seed(int(config.get("seed", 42)))
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True)
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
    if config.get("bf16"):
        import torch

        model_kwargs["torch_dtype"] = torch.bfloat16
    if config.get("flash_attention_2"):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], **model_kwargs)
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
    training_kwargs = dict(
        output_dir=config["output_dir"],
        run_name=config["run_id"],
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_train_epochs"]),
        warmup_ratio=float(config["warmup_ratio"]),
        lr_scheduler_type=config["lr_scheduler_type"],
        weight_decay=float(config["weight_decay"]),
        bf16=bool(config["bf16"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        logging_steps=int(config["logging_steps"]),
        eval_steps=int(config["eval_steps"]),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        save_safetensors=bool(config["save_safetensors"]),
        report_to=config.get("report_to", "none"),
        remove_unused_columns=False,
        seed=int(config.get("seed", 42)),
    )
    evaluation_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    training_kwargs[evaluation_key] = "steps"
    if "include_num_input_tokens_seen" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["include_num_input_tokens_seen"] = True
    arguments = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=CausalChatCollator(tokenizer, int(config["max_seq_length"])),
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
