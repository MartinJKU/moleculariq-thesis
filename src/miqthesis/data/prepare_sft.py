from __future__ import annotations

import argparse
import hashlib
import json
import random
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


def _generated_rows(
    molecules: list[str],
    task_family: str,
    total: int,
    seed: int,
    randomized_probability: float,
    kekulized_probability: float,
) -> Iterable[dict[str, Any]]:
    try:
        from moleculariq_core import MolecularIQD
    except ImportError as exc:
        raise RuntimeError("Install moleculariq-core before preparing training data") from exc
    rng = random.Random(seed)
    generator = MolecularIQD(seed=seed)
    count_properties = list(dict.fromkeys(generator.get_available_count_properties()))
    index_properties = list(dict.fromkeys(generator.get_available_index_properties()))
    if not count_properties or not index_properties:
        raise RuntimeError("moleculariq-core exposed no count/index properties")

    for index in range(total):
        canonical_smiles = molecules[index % len(molecules)]
        smiles, randomized, kekulized = _represent_smiles(
            canonical_smiles, rng, randomized_probability, kekulized_probability
        )
        load = rng.randint(1, 5)
        if task_family == "count":
            properties = rng.sample(count_properties, k=min(load, len(count_properties)))
            question, answer, metadata = generator.generate_count_question(smiles, properties)
            constraints = None
        elif task_family == "index":
            properties = rng.sample(index_properties, k=min(load, len(index_properties)))
            question, answer, metadata = generator.generate_index_question(smiles, properties)
            constraints = None
        elif task_family == "generation":
            properties = rng.sample(count_properties, k=min(load, len(count_properties)))
            constraints = [
                {
                    "property": prop,
                    "operator": "=",
                    "value": generator.compute_property(canonical_smiles, prop),
                }
                for prop in properties
            ]
            question, metadata = generator.generate_constraint_question(constraints)
            answer = {"smiles": canonical_smiles}
        else:
            raise ValueError(f"Unknown task family: {task_family}")
        yield NormalizedExample.from_row(
            {
                "uid": _uid(seed, task_family, index, canonical_smiles, properties),
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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with jsonlines.open(path, mode="w") as writer:
        for row in rows:
            writer.write(row)
            count += 1
    return count


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

    family_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family in ("count", "index", "generation"):
        sizes = config["sizes"][family]
        family_rows[family] = {
            "train": list(
                _generated_rows(
                    train_molecules,
                    family,
                    int(sizes["train"]),
                    seed + len(family),
                    random_p,
                    kekule_p,
                )
            ),
            "val": list(
                _generated_rows(
                    validation_molecules,
                    family,
                    int(sizes["val"]),
                    seed + 1_000 + len(family),
                    random_p,
                    kekule_p,
                )
            ),
        }
        for split in ("train", "val"):
            name = f"sft_{family}_{split}.jsonl"
            counts[name] = _write_jsonl(output_dir / name, family_rows[family][split])

    multitask_sizes = config["sizes"]["multitask"]
    for split in ("train", "val"):
        target_size = int(multitask_sizes[split])
        combined = sum((family_rows[family][split] for family in family_rows), [])
        rng = random.Random(seed + (200 if split == "train" else 201))
        rng.shuffle(combined)
        if target_size > len(combined):
            combined = [combined[index % len(combined)] for index in range(target_size)]
        else:
            combined = combined[:target_size]
        name = f"sft_multitask_{split}.jsonl"
        counts[name] = _write_jsonl(output_dir / name, combined)

    (output_dir / "sft_manifest.json").write_text(
        json.dumps(
            {
                "config": config,
                "counts": counts,
                "train_molecules": len(train_molecules),
                "validation_molecules": len(validation_molecules),
                "molecule_split_overlap": 0,
            },
            indent=2,
        )
        + "\n",
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
