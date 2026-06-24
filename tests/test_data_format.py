import json

import pytest

from miqthesis.data.prepare_sft import split_molecules
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
