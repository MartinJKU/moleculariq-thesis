from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from miqthesis.constants import CONTROLLED_GENERATION_CONFIG, IM_END_TOKEN_ID


def _full_weights_present(path: Path) -> bool:
    direct = any(
        (path / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
    )
    sharded = any(path.glob("model-*.safetensors")) or any(
        path.glob("pytorch_model-*.bin")
    )
    return direct or sharded


def _tokenizer_present(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    )


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _assert_stop_tokens(generation_config: dict[str, Any]) -> None:
    """Reject a generation config that cannot stop the assistant turn under vLLM."""
    eos = generation_config.get("eos_token_id")
    eos_ids = eos if isinstance(eos, list) else [eos]
    if IM_END_TOKEN_ID not in eos_ids:
        raise ValueError(
            f"generation_config eos_token_id={eos!r} does not include the chat "
            f"turn-end token <|im_end|> ({IM_END_TOKEN_ID}). A vLLM evaluation would "
            "not stop at the end of the assistant turn and would over-generate, "
            "breaking answer extraction."
        )


def prepare_checkpoint(
    source: str | Path,
    destination: str | Path,
    generation_config: dict[str, Any] | None = None,
    validate_load: bool = True,
) -> None:
    source, destination = Path(source), Path(destination)
    gen_config = generation_config or CONTROLLED_GENERATION_CONFIG
    _assert_stop_tokens(gen_config)
    adapter_files = [
        source / "adapter_config.json",
        source / "adapter_model.safetensors",
        source / "adapter_model.bin",
    ]
    if any(path.exists() for path in adapter_files):
        raise ValueError(f"Adapter checkpoint rejected by full-parameter invariant: {source}")
    if not _full_weights_present(source):
        raise ValueError(f"No full model weights found in {source}")
    if not _tokenizer_present(source):
        raise ValueError(f"No tokenizer files found in {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    (destination / "generation_config.json").write_bytes(_canonical_bytes(gen_config))
    if validate_load:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        AutoTokenizer.from_pretrained(destination, local_files_only=True)
        AutoModelForCausalLM.from_pretrained(
            destination,
            device_map="cpu",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )


def validate_consistency(eval_root: str | Path) -> dict[str, str]:
    eval_root = Path(eval_root)
    checkpoints = sorted(path for path in eval_root.iterdir() if path.is_dir())
    if not checkpoints:
        raise ValueError(f"No evaluation checkpoints in {eval_root}")
    generation_hashes: dict[str, str] = {}
    chat_templates: dict[str, str] = {}
    for path in checkpoints:
        generation = path / "generation_config.json"
        if not generation.exists():
            raise ValueError(f"Missing generation config: {generation}")
        _assert_stop_tokens(json.loads(generation.read_text(encoding="utf-8")))
        generation_hashes[path.name] = hashlib.sha256(generation.read_bytes()).hexdigest()
        tokenizer_config = path / "tokenizer_config.json"
        payload = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        chat_templates[path.name] = str(payload.get("chat_template", ""))
    if len(set(generation_hashes.values())) != 1:
        raise ValueError(f"Protocol A generation config mismatch: {generation_hashes}")
    if len(set(chat_templates.values())) != 1:
        raise ValueError("Protocol A chat template mismatch across evaluation checkpoints")
    return generation_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--destination")
    parser.add_argument("--eval_root", default="checkpoints/eval")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--skip_load_validation", action="store_true")
    parser.add_argument(
        "--skip_consistency",
        action="store_true",
        help=(
            "Stage a single checkpoint (e.g. a mid-training step) without requiring "
            "every checkpoint under --eval_root to share one generation config and "
            "chat template. The eos-token guard still runs."
        ),
    )
    args = parser.parse_args()
    if not args.validate_only:
        if not args.source or not args.destination:
            parser.error("--source and --destination are required unless --validate_only")
        prepare_checkpoint(
            args.source,
            args.destination,
            validate_load=not args.skip_load_validation,
        )
        if args.skip_consistency:
            return
    validate_consistency(args.eval_root)


if __name__ == "__main__":
    main()
