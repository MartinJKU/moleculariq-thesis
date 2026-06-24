import pytest

pytest.importorskip("rdkit")

from miqthesis.data.splits import leakage_filter


def test_canonical_smiles_leakage_is_removed():
    benchmark = [{"original_smiles": "C(C)O"}]
    train = [
        {"uid": "same", "original_smiles": "CCO"},
        {"uid": "other", "original_smiles": "c1ccccc1"},
    ]
    kept, report = leakage_filter(train, benchmark)
    assert [row["uid"] for row in kept] == ["other"]
    assert report["removed_training_rows"] == 1

