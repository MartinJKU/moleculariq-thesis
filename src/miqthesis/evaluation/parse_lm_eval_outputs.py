from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from miqthesis.data.schemas import NormalizedExample, first_present, stable_uid
from miqthesis.evaluation.verify_outputs import verify_prediction
from miqthesis.training.rewards import extract_answer_block


def _flatten_responses(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_responses(item))
        return flattened
    if isinstance(value, dict):
        for key in ("text", "content", "response"):
            if key in value:
                return _flatten_responses(value[key])
    return []


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def find_official_metrics(run_dir: str | Path) -> dict[str, Any]:
    candidates = sorted(
        path
        for path in Path(run_dir).rglob("*.json")
        if "result" in path.name.lower() and path.name != "run_manifest.json"
    )
    for path in reversed(candidates):
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results", payload)
        if isinstance(results, dict):
            for task, metrics in results.items():
                if isinstance(metrics, dict) and any(
                    key.split(",")[0] == "avg_accuracy" for key in metrics
                ):
                    normalized = {
                        key.split(",")[0]: value
                        for key, value in metrics.items()
                        if not key.endswith("_stderr,none")
                    }
                    normalized["task"] = task
                    normalized["source_file"] = str(path)
                    return normalized
    return {}


def _sample_files(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.rglob("*.jsonl")
        if "sample" in path.name.lower() or "samples" in str(path.parent).lower()
    )


def parse_run(
    run_dir: str | Path,
    model_id: str,
    protocol: str,
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    run_dir = Path(run_dir)
    metadata_by_uid: dict[str, dict[str, Any]] = {}
    if benchmark is not None:
        for row in benchmark.to_dict("records"):
            metadata_by_uid[stable_uid(row)] = row
            explicit = first_present(row, "uid", "id", "question_id")
            if explicit is not None:
                metadata_by_uid[str(explicit)] = row

    records: list[dict[str, Any]] = []
    for sample_file in _sample_files(run_dir):
        for sample_index, sample in enumerate(_read_jsonl(sample_file)):
            doc = dict(sample.get("doc") or sample.get("document") or {})
            uid = str(
                first_present(
                    doc,
                    "uid",
                    "id",
                    "question_id",
                    default=sample.get("doc_id", sample_index),
                )
            )
            merged = {**metadata_by_uid.get(uid, {}), **doc}
            normalized = NormalizedExample.from_row({**merged, "uid": uid})
            responses = _flatten_responses(
                first_present(sample, "resps", "responses", "filtered_resps", default=[])
            )
            if not responses:
                continue
            attempt_rows: list[dict[str, Any]] = []
            for repeat_index, response in enumerate(responses):
                verification = verify_prediction(
                    response,
                    normalized.target,
                    normalized.task_type,
                    normalized.constraints,
                )
                attempt_rows.append(
                    {
                        "model_id": model_id,
                        "run_id": run_dir.name,
                        "protocol": protocol.upper(),
                        "uid": normalized.uid,
                        "repeat_index": repeat_index,
                        "task_type": normalized.task_type,
                        "features": normalized.features,
                        "complexity_bin": normalized.complexity_bin,
                        "multi_task_load": normalized.multi_task_load,
                        "is_randomized": normalized.is_randomized,
                        "is_kekulized": normalized.is_kekulized,
                        "question": normalized.question,
                        "target": normalized.target,
                        "prediction_raw": response,
                        "prediction_answer_block": extract_answer_block(response),
                        "prediction_json": json.dumps(
                            verification["prediction_json"], ensure_ascii=False
                        )
                        if verification["prediction_json"] is not None
                        else None,
                        "exact": verification["exact"],
                        "field_accuracy": verification["field_accuracy"],
                        "valid_json": verification["valid_json"],
                        "valid_smiles": verification["valid_smiles"],
                        "constraint_satisfaction": verification[
                            "constraint_satisfaction"
                        ],
                        "latency": sample.get("latency"),
                        "num_prompt_tokens": sample.get("num_prompt_tokens"),
                        "num_completion_tokens": sample.get("num_completion_tokens"),
                        "source_file": str(sample_file),
                    }
                )
            pass_1 = float(attempt_rows[0]["exact"] > 0)
            pass_3 = float(any(row["exact"] > 0 for row in attempt_rows[:3]))
            for row in attempt_rows:
                row["pass_at_1_component"] = pass_1
                row["pass_at_3_component"] = pass_3
            records.extend(attempt_rows)
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--benchmark")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--protocol", choices=["A", "A_DETERMINISTIC", "B"])
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    benchmark = pd.read_parquet(args.benchmark) if args.benchmark else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocols = [args.protocol] if args.protocol else ["A", "A_DETERMINISTIC", "B"]
    for protocol in protocols:
        protocol_dir = input_dir / f"protocol_{protocol.lower()}"
        if not protocol_dir.exists():
            continue
        for run_dir in sorted(path for path in protocol_dir.iterdir() if path.is_dir()):
            frame = parse_run(run_dir, run_dir.name, protocol, benchmark)
            if frame.empty:
                print(f"No logged samples found for {run_dir}")
                continue
            frame.to_parquet(
                output_dir / f"{protocol.lower()}_{run_dir.name}_samples.parquet",
                index=False,
            )
            metrics = find_official_metrics(run_dir)
            metrics.update(
                {
                    "model_id": run_dir.name,
                    "protocol": protocol,
                    "derived_pass_at_1": frame.groupby("uid")[
                        "pass_at_1_component"
                    ].first().mean(),
                    "derived_pass_at_3": frame.groupby("uid")[
                        "pass_at_3_component"
                    ].first().mean(),
                }
            )
            (output_dir / f"{protocol.lower()}_{run_dir.name}_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
