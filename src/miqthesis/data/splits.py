from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any


def canonicalize_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for leakage-safe SMILES canonicalization") from exc
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def deterministic_bucket(uid: str, seed: int = 42, buckets: int = 10_000) -> int:
    digest = hashlib.sha256(f"{seed}:{uid}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % buckets


def deterministic_split(
    rows: Iterable[Mapping[str, Any]],
    validation_fraction: float = 0.04,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    cutoff = int(validation_fraction * 10_000)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        item = dict(row)
        uid = str(item["uid"])
        if uid in seen:
            raise ValueError(f"Duplicate uid before split: {uid}")
        seen.add(uid)
        destination = validation if deterministic_bucket(uid, seed) < cutoff else train
        destination.append(item)
    return train, validation


def assert_disjoint_uids(train: Iterable[Mapping[str, Any]], val: Iterable[Mapping[str, Any]]) -> None:
    train_ids = {str(row["uid"]) for row in train}
    val_ids = {str(row["uid"]) for row in val}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Train/validation UID overlap: {sorted(overlap)[:10]}")


def leakage_filter(
    train_rows: Iterable[Mapping[str, Any]],
    benchmark_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark_canonical = {
        canonical
        for row in benchmark_rows
        if (canonical := canonicalize_smiles(
            row.get("original_smiles") or row.get("smiles")
        ))
    }
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    invalid_smiles = 0
    for row in train_rows:
        item = dict(row)
        canonical = canonicalize_smiles(
            item.get("original_smiles") or item.get("smiles")
        )
        item["canonical_smiles"] = canonical
        if canonical is None:
            invalid_smiles += 1
        if canonical and canonical in benchmark_canonical:
            removed.append(item)
        else:
            kept.append(item)
    report = {
        "benchmark_unique_canonical_smiles": len(benchmark_canonical),
        "input_training_rows": len(kept) + len(removed),
        "kept_training_rows": len(kept),
        "removed_training_rows": len(removed),
        "invalid_training_smiles": invalid_smiles,
        "removed_uids": [str(row.get("uid", "")) for row in removed],
    }
    return kept, report

