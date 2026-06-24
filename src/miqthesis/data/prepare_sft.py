from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable

import jsonlines
import pandas as pd

from miqthesis.data.schemas import NormalizedExample
from miqthesis.data.splits import deterministic_bucket
from miqthesis.training.utils import load_yaml


def _complexity_bin(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem import GraphDescriptors

        molecule = Chem.MolFromSmiles(smiles)
        complexity = GraphDescriptors.BertzCT(molecule) if molecule else 0
    except ImportError:
        complexity = 0
    if complexity < 250:
        return "0-250"
    if complexity < 1000:
        return "250-1000"
    return "1000+"


def _represent_smiles(
    smiles: str,
    rng: random.Random,
    randomized_probability: float,
    kekulized_probability: float,
) -> tuple[str, bool, bool]:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required to prepare MolecularIQ data") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES in filtered pool: {smiles}")
    randomized = rng.random() < randomized_probability
    kekulized = rng.random() < kekulized_probability
    rendered = Chem.MolToSmiles(
        molecule,
        canonical=not randomized,
        doRandom=randomized,
        kekuleSmiles=kekulized,
        isomericSmiles=True,
    )
    return rendered, randomized, kekulized


def _uid(*parts: Any) -> str:
    material = "|".join(map(str, parts))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _record_solver_failure(
    diagnostics: dict[str, Any],
    canonical_smiles: str,
    property_name: str,
    error: Exception,
) -> None:
    diagnostics["solver_failures"] += 1
    diagnostics["failure_types"][type(error).__name__] += 1
    diagnostics["failed_properties"][property_name] += 1
    diagnostics["failed_molecules"].add(canonical_smiles)


def _select_safe_properties(
    generator: Any,
    evaluation_smiles: str,
    canonical_smiles: str,
    property_pool: list[str],
    requested: int,
    rng: random.Random,
    failed_pairs: set[tuple[str, str]],
    diagnostics: dict[str, Any],
) -> list[tuple[str, Any]]:
    """Select properties whose upstream symbolic solver succeeds for this molecule."""
    selected: list[tuple[str, Any]] = []
    tried: set[str] = set()
    draws = 0
    max_draws = max(len(property_pool) * 4, requested * 8)
    while (
        len(selected) < requested
        and len(tried) < len(property_pool)
        and draws < max_draws
    ):
        draws += 1
        property_name = rng.choice(property_pool)
        if property_name in tried:
            continue
        tried.add(property_name)
        failure_key = (canonical_smiles, property_name)
        if failure_key in failed_pairs:
            continue
        try:
            value = generator.compute_property(evaluation_smiles, property_name)
        except Exception as error:
            # moleculariq-core contains recursive graph algorithms that can raise
            # RecursionError on pathological molecules. Other property-specific
            # solver failures should be isolated in exactly the same way.
            failed_pairs.add(failure_key)
            _record_solver_failure(
                diagnostics, canonical_smiles, property_name, error
            )
            continue
        selected.append((property_name, value))
    return selected


def _generated_rows(
    molecules: list[str],
    task_family: str,
    total: int,
    seed: int,
    randomized_probability: float,
    kekulized_probability: float,
    diagnostics: dict[str, Any] | None = None,
) -> Iterable[dict[str, Any]]:
    try:
        from moleculariq_core import MolecularIQD
    except ImportError as exc:
        raise RuntimeError("Install moleculariq-core before preparing training data") from exc
    rng = random.Random(seed)
    # The property cache grows with every molecule/property pair. That is useful
    # for interactive reuse, but a large one-pass corpus build can retain hundreds
    # of thousands of RDKit-derived objects and be killed by a login-node cgroup.
    generator = MolecularIQD(seed=seed, cache_properties=False)
    count_properties = list(dict.fromkeys(generator.get_available_count_properties()))
    index_properties = list(dict.fromkeys(generator.get_available_index_properties()))
    if not count_properties or not index_properties:
        raise RuntimeError("moleculariq-core exposed no count/index properties")

    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault("solver_failures", 0)
    diagnostics.setdefault("row_retries", 0)
    diagnostics.setdefault("failure_types", Counter())
    diagnostics.setdefault("failed_properties", Counter())
    diagnostics.setdefault("failed_molecules", set())
    failed_pairs: set[tuple[str, str]] = set()
    produced = 0
    attempts = 0
    max_attempts = max(total * 20, len(molecules) * 2, 1_000)

    while produced < total:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could only generate {produced:,}/{total:,} {task_family} examples "
                f"after {attempts:,} attempts. Solver failures: "
                f"{diagnostics['solver_failures']:,}. See sft_manifest.json diagnostics."
            )
        canonical_smiles = molecules[attempts % len(molecules)]
        attempts += 1
        smiles, randomized, kekulized = _represent_smiles(
            canonical_smiles, rng, randomized_probability, kekulized_probability
        )
        load = rng.randint(1, 5)
        if task_family == "count":
            selected = _select_safe_properties(
                generator,
                smiles,
                canonical_smiles,
                count_properties,
                min(load, len(count_properties)),
                rng,
                failed_pairs,
                diagnostics,
            )
            properties = [property_name for property_name, _ in selected]
            if len(properties) < load:
                diagnostics["row_retries"] += 1
                continue
            try:
                question, answer, metadata = generator.generate_count_question(
                    smiles, properties
                )
            except Exception as error:
                diagnostics["row_retries"] += 1
                for property_name in properties:
                    failed_pairs.add((canonical_smiles, property_name))
                    _record_solver_failure(
                        diagnostics, canonical_smiles, property_name, error
                    )
                continue
            constraints = None
        elif task_family == "index":
            selected = _select_safe_properties(
                generator,
                smiles,
                canonical_smiles,
                index_properties,
                min(load, len(index_properties)),
                rng,
                failed_pairs,
                diagnostics,
            )
            properties = [property_name for property_name, _ in selected]
            if len(properties) < load:
                diagnostics["row_retries"] += 1
                continue
            try:
                question, answer, metadata = generator.generate_index_question(
                    smiles, properties
                )
            except Exception as error:
                diagnostics["row_retries"] += 1
                for property_name in properties:
                    failed_pairs.add((canonical_smiles, property_name))
                    _record_solver_failure(
                        diagnostics, canonical_smiles, property_name, error
                    )
                continue
            constraints = None
        elif task_family == "generation":
            selected = _select_safe_properties(
                generator,
                canonical_smiles,
                canonical_smiles,
                count_properties,
                min(load, len(count_properties)),
                rng,
                failed_pairs,
                diagnostics,
            )
            properties = [property_name for property_name, _ in selected]
            if len(properties) < load:
                diagnostics["row_retries"] += 1
                continue
            constraints = [
                {
                    "property": property_name,
                    "operator": "=",
                    "value": value,
                }
                for property_name, value in selected
            ]
            try:
                question, metadata = generator.generate_constraint_question(constraints)
            except Exception as error:
                diagnostics["row_retries"] += 1
                for property_name in properties:
                    failed_pairs.add((canonical_smiles, property_name))
                    _record_solver_failure(
                        diagnostics, canonical_smiles, property_name, error
                    )
                continue
            answer = {"smiles": canonical_smiles}
        else:
            raise ValueError(f"Unknown task family: {task_family}")
        yield NormalizedExample.from_row(
            {
                "uid": _uid(seed, task_family, produced, canonical_smiles, properties),
                "question": question,
                "target": answer,
                "task_type": metadata["task_type"],
                "features": ",".join(properties),
                "constraints": constraints,
                "original_smiles": canonical_smiles,
                "complexity_bin": _complexity_bin(canonical_smiles),
                "multi_task_load": len(properties),
                "is_randomized": randomized,
                "is_kekulized": kekulized,
                "generator_metadata": metadata,
            }
        ).to_sft()
        produced += 1


def _write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    expected: int | None = None,
    progress_every: int = 1_000,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with jsonlines.open(path, mode="w") as writer:
        for row in rows:
            writer.write(row)
            count += 1
            if progress_every and count % progress_every == 0:
                suffix = f"/{expected:,}" if expected is not None else ""
                print(f"{path.name}: {count:,}{suffix}", flush=True)
    print(f"{path.name}: complete ({count:,} rows)", flush=True)
    return count


def _interleave_jsonl(
    source_paths: list[Path],
    destination: Path,
    target_size: int,
    seed: int,
    source_counts: dict[str, int],
) -> int:
    """Write a balanced, bounded-memory interleave of family JSONL files."""
    available = sum(source_counts[path.name] for path in source_paths)
    if target_size > available:
        raise ValueError(
            f"Requested {target_size:,} rows for {destination.name}, but only "
            f"{available:,} source rows exist. Refusing to duplicate UIDs."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = 0
    with ExitStack() as stack:
        readers = {
            path.name: iter(stack.enter_context(jsonlines.open(path)))
            for path in source_paths
        }
        with jsonlines.open(destination, mode="w") as writer:
            while written < target_size and readers:
                names = list(readers)
                rng.shuffle(names)
                for name in names:
                    if written >= target_size:
                        break
                    try:
                        row = next(readers[name])
                    except StopIteration:
                        del readers[name]
                        continue
                    writer.write(row)
                    written += 1
                    if written % 1_000 == 0:
                        print(
                            f"{destination.name}: {written:,}/{target_size:,}",
                            flush=True,
                        )
    if written != target_size:
        raise RuntimeError(
            f"Only wrote {written:,}/{target_size:,} rows to {destination}"
        )
    print(f"{destination.name}: complete ({written:,} rows)", flush=True)
    return written


def split_molecules(
    molecules: list[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    cutoff = int(validation_fraction * 10_000)
    train = [
        smiles
        for smiles in molecules
        if deterministic_bucket(smiles, seed) >= cutoff
    ]
    validation = [
        smiles
        for smiles in molecules
        if deterministic_bucket(smiles, seed) < cutoff
    ]
    if not train or not validation:
        raise ValueError(
            "Molecule-level split produced an empty partition; use a larger pool "
            "or adjust validation_fraction."
        )
    if set(train) & set(validation):
        raise RuntimeError("Molecule-level train/validation overlap")
    return train, validation


def prepare(config_path: str | Path) -> dict[str, int]:
    config = load_yaml(config_path)
    pool = pd.read_parquet(config["input_train_pool"])
    smiles_column = (
        "canonical_smiles" if "canonical_smiles" in pool.columns else "original_smiles"
    )
    molecules = [str(value) for value in pool[smiles_column].dropna().unique()]
    if not molecules:
        raise RuntimeError("No molecules available for SFT generation")
    output_dir = Path(config["output_dir"])
    seed = int(config.get("seed", 42))
    train_molecules, validation_molecules = split_molecules(
        molecules, float(config.get("validation_fraction", 0.04)), seed
    )
    representation = config.get("representation", {})
    random_p = float(representation.get("randomized_probability", 0.25))
    kekule_p = float(representation.get("kekulized_probability", 0.25))
    counts: dict[str, int] = {}

    generation_diagnostics: dict[str, dict[str, Any]] = {}
    for family in ("count", "index", "generation"):
        sizes = config["sizes"][family]
        train_diagnostics: dict[str, Any] = {}
        validation_diagnostics: dict[str, Any] = {}
        split_specs = {
            "train": (
                train_molecules,
                int(sizes["train"]),
                seed + len(family),
                train_diagnostics,
            ),
            "val": (
                validation_molecules,
                int(sizes["val"]),
                seed + 1_000 + len(family),
                validation_diagnostics,
            ),
        }
        for split, (split_molecule_pool, split_size, split_seed, diagnostics) in (
            split_specs.items()
        ):
            name = f"sft_{family}_{split}.jsonl"
            counts[name] = _write_jsonl(
                output_dir / name,
                _generated_rows(
                    split_molecule_pool,
                    family,
                    split_size,
                    split_seed,
                    random_p,
                    kekule_p,
                    diagnostics,
                ),
                expected=split_size,
            )
        generation_diagnostics[family] = {
            "train": train_diagnostics,
            "val": validation_diagnostics,
        }

    multitask_sizes = config["sizes"]["multitask"]
    for split in ("train", "val"):
        target_size = int(multitask_sizes[split])
        name = f"sft_multitask_{split}.jsonl"
        source_paths = [
            output_dir / f"sft_{family}_{split}.jsonl"
            for family in ("count", "index", "generation")
        ]
        counts[name] = _interleave_jsonl(
            source_paths,
            output_dir / name,
            target_size,
            seed + (200 if split == "train" else 201),
            counts,
        )

    (output_dir / "sft_manifest.json").write_text(
        json.dumps(
            {
                "config": config,
                "counts": counts,
                "train_molecules": len(train_molecules),
                "validation_molecules": len(validation_molecules),
                "molecule_split_overlap": 0,
                "generation_diagnostics": _json_safe_diagnostics(
                    generation_diagnostics
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return counts


def _json_safe_diagnostics(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value.most_common())
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {
            key: _json_safe_diagnostics(item)
            for key, item in value.items()
        }
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    for name, count in prepare(args.config).items():
        print(f"{name}: {count:,}")


if __name__ == "__main__":
    main()
