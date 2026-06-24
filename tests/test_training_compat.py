from __future__ import annotations

import pytest

from miqthesis.training.utils import transformers_dtype_kwargs


def test_transformers_v4_uses_legacy_torch_dtype_name() -> None:
    assert transformers_dtype_kwargs("auto", version="4.56.2") == {
        "torch_dtype": "auto"
    }


def test_transformers_v5_uses_dtype_name() -> None:
    assert transformers_dtype_kwargs("auto", version="5.0.0") == {"dtype": "auto"}


def test_invalid_transformers_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="Cannot parse Transformers version"):
        transformers_dtype_kwargs("auto", version="dev")
