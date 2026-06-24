from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODELS = {
    "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct",
}

REQUIRED_FILES = {
    "config.json",
    "tokenizer_config.json",
}


def validate_model_directory(path: str | Path) -> None:
    path = Path(path)
    missing = sorted(name for name in REQUIRED_FILES if not (path / name).exists())
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    if missing or not has_weights:
        details = []
        if missing:
            details.append(f"missing files: {missing}")
        if not has_weights:
            details.append("missing model weights")
        raise RuntimeError(f"Incomplete model snapshot at {path}: {', '.join(details)}")


def download_models(
    model_root: str | Path,
    model_ids: list[str] | None = None,
) -> dict[str, str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required for model staging. "
            "Run: python -m pip install -e '.[train,chem]'"
        ) from exc
    model_root = Path(model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    selected = model_ids or list(DEFAULT_MODELS)
    downloaded: dict[str, str] = {}
    for model_id in selected:
        destination_name = DEFAULT_MODELS.get(model_id, model_id.rsplit("/", 1)[-1])
        destination = model_root / destination_name
        print(f"Staging {model_id} -> {destination}", flush=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=destination,
        )
        validate_model_directory(destination)
        downloaded[model_id] = str(destination)
    manifest = model_root / "download_manifest.json"
    manifest.write_text(
        json.dumps(downloaded, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--model-id", action="append", dest="model_ids")
    args = parser.parse_args()
    download_models(args.model_root, args.model_ids)


if __name__ == "__main__":
    main()
