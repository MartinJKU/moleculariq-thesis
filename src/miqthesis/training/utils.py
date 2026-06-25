from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default or "")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _expand_env(data)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def require_cuda(context: str = "Training") -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"{context} requires PyTorch with CUDA support") from exc

    if torch.cuda.is_available():
        return

    torch_version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    raise RuntimeError(
        f"{context} requires a working CUDA GPU, but PyTorch cannot initialize CUDA "
        f"(torch={torch_version}, torch CUDA build={cuda_version or 'CPU-only'}). "
        "Run GPU jobs on a Leonardo boost_usr_prod node. If the torch CUDA build "
        "shows CPU-only, rebuild the environment with "
        "`MIQ_INSTALL_DEPS=1 source scripts/00_setup_env.sh`."
    )


def transformers_dtype_kwargs(dtype: Any, version: str | None = None) -> dict[str, Any]:
    if version is None:
        import transformers

        version = transformers.__version__
    try:
        major = int(version.split(".", maxsplit=1)[0])
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Cannot parse Transformers version: {version!r}") from exc
    return {"dtype" if major >= 5 else "torch_dtype": dtype}


def assert_full_parameter_config(config: dict[str, Any]) -> None:
    if config.get("full_parameter_finetuning") is not True:
        raise ValueError("full_parameter_finetuning must be explicitly true")
    forbidden = {
        "peft_config",
        "lora_r",
        "lora_alpha",
        "adapter",
        "load_in_4bit",
        "load_in_8bit",
        "freeze_backbone",
    }
    present = sorted(key for key in forbidden if config.get(key))
    if present:
        raise ValueError(f"Full-parameter run contains forbidden options: {present}")


def require_local_model(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_dir() or not (path / "config.json").exists():
        raise FileNotFoundError(
            f"Local model snapshot not found at {path}. "
            "On a Leonardo login node run: bash scripts/01_download_models.sh"
        )
    return path
