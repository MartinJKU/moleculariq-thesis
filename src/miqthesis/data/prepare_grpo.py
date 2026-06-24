from __future__ import annotations

import argparse
import json
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


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_systematic_sample(path: Path, source: Path, target_size: int) -> int:
    """Select exactly target_size evenly spaced rows without loading the file."""
    source_size = _count_rows(source)
    if target_size > source_size:
        raise ValueError(
            f"Requested {target_size:,} GRPO rows from only {source_size:,} SFT rows"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with jsonlines.open(source) as reader, jsonlines.open(path, mode="w") as writer:
        for index, row in enumerate(reader):
            previous = index * target_size // source_size
            current = (index + 1) * target_size // source_size
            if current > previous:
                writer.write(_to_grpo(dict(row)))
                written += 1
                if written % 1_000 == 0:
                    print(
                        f"{path.name}: {written:,}/{target_size:,}",
                        flush=True,
                    )
    if written != target_size:
        raise RuntimeError(f"Only wrote {written:,}/{target_size:,} rows to {path}")
    print(f"{path.name}: complete ({written:,} rows)", flush=True)
    return written


def prepare(config_path: str | Path) -> dict[str, int]:
    config = load_yaml(config_path)
    output_dir = Path(config["output_dir"])
    requested = config["sizes"]["grpo"]
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        source = output_dir / f"sft_multitask_{split}.jsonl"
        destination = output_dir / f"grpo_multitask_{split}.jsonl"
        counts[destination.name] = _write_systematic_sample(
            destination, source, int(requested[split])
        )
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
