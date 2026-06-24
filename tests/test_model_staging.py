import pytest

from miqthesis.data.download_models import validate_model_directory
from miqthesis.training.utils import require_local_model


def test_local_model_preflight_has_actionable_error(tmp_path):
    missing = tmp_path / "missing-model"
    with pytest.raises(FileNotFoundError, match="01_download_models.sh"):
        require_local_model(missing)


def test_model_snapshot_validation_requires_config_tokenizer_and_weights(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"placeholder")
    validate_model_directory(model)
