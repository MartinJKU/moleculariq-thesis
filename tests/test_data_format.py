import json
import random
from collections import Counter

import pytest

from miqthesis.data.prepare_sft import _select_safe_properties, split_molecules
from miqthesis.data.schemas import NormalizedExample
from miqthesis.data.splits import assert_disjoint_uids, deterministic_split


def test_sft_example_has_required_chat_and_json():
    example = NormalizedExample.from_row(
        {
            "uid": "x",
            "question": "Count rings.",
            "target": {"ring_count": 1},
            "task_type": "single_count",
            "original_smiles": "C1CC1",
        }
    ).to_sft()
    assert [message["role"] for message in example["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    answer = example["messages"][-1]["content"]
    assert answer.startswith("<answer>") and answer.endswith("</answer>")
    json.loads(answer.removeprefix("<answer>").removesuffix("</answer>"))


def test_split_is_disjoint_and_rejects_duplicates():
    rows = [{"uid": str(index)} for index in range(100)]
    train, val = deterministic_split(rows, validation_fraction=0.2)
    assert_disjoint_uids(train, val)
    assert len(train) + len(val) == 100
    with pytest.raises(ValueError, match="Duplicate uid"):
        deterministic_split([{"uid": "same"}, {"uid": "same"}])


def test_molecule_level_validation_split_is_disjoint():
    molecules = [f"molecule-{index}" for index in range(1000)]
    train, validation = split_molecules(molecules, validation_fraction=0.1, seed=42)
    assert set(train).isdisjoint(validation)
    assert len(train) + len(validation) == len(molecules)


def test_property_selection_skips_upstream_recursion_errors():
    class FakeGenerator:
        def compute_property(self, smiles, property_name):
            if property_name == "longest_carbon_chain_count":
                raise RecursionError("pathological recursive traversal")
            return 2

    diagnostics = {
        "solver_failures": 0,
        "row_retries": 0,
        "failure_types": Counter(),
        "failed_properties": Counter(),
        "failed_molecules": set(),
    }
    failed_pairs = set()
    selected = _select_safe_properties(
        FakeGenerator(),
        "CCO",
        "CCO",
        ["longest_carbon_chain_count", "ring_count", "carbon_atom_count"],
        requested=2,
        rng=random.Random(1),
        failed_pairs=failed_pairs,
        diagnostics=diagnostics,
    )
    assert len(selected) == 2
    assert all(name != "longest_carbon_chain_count" for name, _ in selected)
    assert diagnostics["failure_types"]["RecursionError"] == 1
    assert ("CCO", "longest_carbon_chain_count") in failed_pairs
