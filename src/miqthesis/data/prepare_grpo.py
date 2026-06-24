from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import jsonlines

from miqthesis.training.utils import load_yaml


def _to_grpo(row: dict[str, Any]) -> dict[str, Any]:
    messages = row["messages"]
    if len(messages) < 3:
        raise ValueError("SFT row must contain system, user, and assistant messages")
    output = {key: value for key, value in row.items() if key != "messages"}
    output["prompt"] = messages[:-1]
    return output


def _load(path: Path) -> list[dict[str, Any]]:
    with jsonlines.open(path) as reader:
        return [dict(row) for row in reader]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(rows)


def prepare(config_path: str | Path) -> dict[str, int]:
    config = load_yaml(config_path)
    output_dir = Path(config["output_dir"])
    seed = int(config.get("seed", 42))
    requested = config["sizes"]["grpo"]
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        source = output_dir / f"sft_multitask_{split}.jsonl"
        rows = [_to_grpo(row) for row in _load(source)]
        random.Random(seed + (300 if split == "train" else 301)).shuffle(rows)
        rows = rows[: int(requested[split])]
        destination = output_dir / f"grpo_multitask_{split}.jsonl"
        _write(destination, rows)
        counts[destination.name] = len(rows)
    (output_dir / "grpo_manifest.json").write_text(
        json.dumps({"config": config, "counts": counts}, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    for name, count in prepare(args.config).items():
        print(f"{name}: {count:,}")


if __name__ == "__main__":
    main()

