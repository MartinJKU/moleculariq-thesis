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
    _checkpoint(tmp_path, "a", {"eos_token_id": [151645, 151643], "temperature": 0.7})
    _checkpoint(tmp_path, "b", {"eos_token_id": [151645, 151643], "temperature": 0.7})
    validate_consistency(tmp_path)
    (tmp_path / "b" / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [151645, 151643], "temperature": 0.8}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generation config mismatch"):
        validate_consistency(tmp_path)


def test_adapter_only_checkpoint_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Adapter checkpoint rejected"):
        prepare_checkpoint(source, tmp_path / "destination", validate_load=False)


def test_prepare_rejects_generation_config_without_im_end(tmp_path):
    # A raw training checkpoint inherits eos=[151643] (<|endoftext|> only); staging
    # such a config must fail loudly rather than silently produce a checkpoint that
    # over-generates under vLLM.
    with pytest.raises(ValueError, match="151645"):
        prepare_checkpoint(
            tmp_path / "source",
            tmp_path / "destination",
            generation_config={"eos_token_id": [151643]},
            validate_load=False,
        )


def test_validate_consistency_rejects_missing_im_end(tmp_path):
    _checkpoint(tmp_path, "a", {"eos_token_id": [151643]})
    with pytest.raises(ValueError, match="151645"):
        validate_consistency(tmp_path)
