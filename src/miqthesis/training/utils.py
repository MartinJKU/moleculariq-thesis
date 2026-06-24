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
