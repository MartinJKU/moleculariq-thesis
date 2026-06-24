from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from miqthesis.data.splits import leakage_filter
from miqthesis.training.utils import write_json


def _select_split(dataset: Dataset | DatasetDict, preferred: str) -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset
    if preferred in dataset:
        return dataset[preferred]
    if len(dataset) == 1:
        return next(iter(dataset.values()))
    raise ValueError(f"Dataset has no '{preferred}' split; found {list(dataset)}")


def _save_raw(dataset: Dataset | DatasetDict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(destination))


def download_and_filter(
    benchmark_dataset: str,
    train_pool_dataset: str,
    out_dir: str | Path,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    benchmark = load_dataset(benchmark_dataset, cache_dir=cache_dir)
    train_pool = load_dataset(train_pool_dataset, cache_dir=cache_dir)
    _save_raw(benchmark, raw_dir / "moleculariq_v0.0")
    _save_raw(train_pool, raw_dir / "moleculariq_trainPool")

    benchmark_test = _select_split(benchmark, "test")
    train = _select_split(train_pool, "train")
    benchmark_rows = [dict(row) for row in benchmark_test]
    train_rows = [
        {
            **dict(row),
            "uid": str(row.get("uid", row.get("id", f"pool-{index}"))),
            "original_smiles": row.get("original_smiles", row.get("smiles")),
        }
        for index, row in enumerate(train)
    ]
    filtered, report = leakage_filter(train_rows, benchmark_rows)
    if not filtered:
        raise RuntimeError("Leakage filtering removed the entire training pool")

    pd.DataFrame(filtered).to_parquet(
        processed_dir / "train_pool_filtered.parquet", index=False
    )
    pd.DataFrame(benchmark_rows).to_parquet(
        processed_dir / "benchmark_test.parquet", index=False
    )
    report.update(
        {
            "benchmark_dataset": benchmark_dataset,
            "train_pool_dataset": train_pool_dataset,
            "hard_rule": "No canonical benchmark molecule may appear in training.",
        }
    )
    write_json(processed_dir / "leakage_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dataset", default="ml-jku/moleculariq-v0.0")
    parser.add_argument("--train_pool_dataset", default="ml-jku/moleculariq-trainPool")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--cache_dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = download_and_filter(
        args.benchmark_dataset, args.train_pool_dataset, args.out_dir, args.cache_dir
    )
    print(
        f"Kept {report['kept_training_rows']:,} molecules; "
        f"removed {report['removed_training_rows']:,} benchmark overlaps."
    )


if __name__ == "__main__":
    main()

