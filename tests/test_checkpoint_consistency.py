import json

import pytest

from miqthesis.training.prepare_checkpoint import prepare_checkpoint, validate_consistency


def _checkpoint(root, name, generation, template="same"):
    path = root / name
    path.mkdir()
    (path / "generation_config.json").write_text(
        json.dumps(generation, sort_keys=True), encoding="utf-8"
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template}), encoding="utf-8"
    )


def test_protocol_a_checkpoint_configs_must_be_identical(tmp_path):
    _checkpoint(tmp_path, "a", {"temperature": 0.7})
    _checkpoint(tmp_path, "b", {"temperature": 0.7})
    validate_consistency(tmp_path)
    (tmp_path / "b" / "generation_config.json").write_text(
        json.dumps({"temperature": 0.8}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="generation config mismatch"):
        validate_consistency(tmp_path)


def test_adapter_only_checkpoint_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Adapter checkpoint rejected"):
        prepare_checkpoint(source, tmp_path / "destination", validate_load=False)
