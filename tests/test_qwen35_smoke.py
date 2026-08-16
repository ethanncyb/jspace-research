"""Optional local-model smoke test; never downloads weights."""

import os

import pytest

from model_hooks import (
    DEFAULT_FULL_ATTENTION_LAYERS,
    hidden_size,
    load_model_and_tokenizer,
    validate_layer_layout,
)


@pytest.mark.slow
def test_locally_available_qwen35_layout() -> None:
    path = os.environ.get("QWEN35_LOCAL_PATH")
    if not path:
        pytest.skip("set QWEN35_LOCAL_PATH to exercise a local Qwen 3.5 checkpoint")
    model, _tokenizer, _policy = load_model_and_tokenizer(
        path, device="cpu", dtype="float32", local_files_only=True
    )
    assert validate_layer_layout(model, DEFAULT_FULL_ATTENTION_LAYERS)
    assert hidden_size(model) in {2560, 4096}
