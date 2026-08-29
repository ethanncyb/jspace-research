from __future__ import annotations

import pytest
import torch
from torch import nn

from jspace_research.model import HuggingFaceModelAdapter, _validate_primary_gpu_placement
from jspace_research.phase1.adapters import validate_lens_for_layers, validate_model_lens


class FakeModel:
    hidden_width = 3
    number_layers = 4


class FakeLens:
    hidden_width = 3
    source_layers = (1, 2)

    def jacobian(self, layer: int) -> torch.Tensor:
        assert layer in self.source_layers
        return torch.eye(3)


def test_model_lens_compatibility_is_explicitly_validated() -> None:
    validate_model_lens(FakeModel(), FakeLens())  # type: ignore[arg-type]

    incompatible = FakeLens()
    incompatible.hidden_width = 4
    with pytest.raises(RuntimeError, match="width mismatch"):
        validate_model_lens(FakeModel(), incompatible)  # type: ignore[arg-type]


def test_cached_layers_must_be_fitted_by_lens() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        validate_lens_for_layers(FakeLens(), 3, [0])  # type: ignore[arg-type]


def test_model_placement_rejects_cpu_disk_and_multiple_gpus() -> None:
    class PlacedModel:
        hf_device_map = {"": 0}

    _validate_primary_gpu_placement(PlacedModel())

    for device_map in ({"": "cpu"}, {"layer": "disk"}, {"a": 0, "b": 1}):
        model = PlacedModel()
        model.hf_device_map = device_map
        with pytest.raises(RuntimeError, match="fit entirely on CUDA GPU 0"):
            _validate_primary_gpu_placement(model)


def test_model_placement_must_be_verifiable() -> None:
    with pytest.raises(RuntimeError, match="Could not verify"):
        _validate_primary_gpu_placement(object())


class RecordingBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, str]:
        return hidden + 1, "cache"


class FakeGenerationModel:
    def __init__(self, block: RecordingBlock) -> None:
        self.block = block
        self.prefill: torch.Tensor | None = None
        self.decode: torch.Tensor | None = None

    def generate(self, *, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        hidden = torch.zeros((1, input_ids.shape[-1], 3))
        self.prefill = self.block(hidden)[0].detach().clone()
        self.decode = self.block(torch.zeros((1, 1, 3)))[0].detach().clone()
        return torch.cat([input_ids, torch.tensor([[9]], device=input_ids.device)], dim=-1)


class FakeLensModel:
    def __init__(self, block: RecordingBlock) -> None:
        self.layers = nn.ModuleList([block])
        self.n_layers = 1
        self.d_model = 3
        self.input_device = torch.device("cpu")


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2


def make_generation_adapter() -> tuple[HuggingFaceModelAdapter, FakeGenerationModel]:
    block = RecordingBlock()
    hf_model = FakeGenerationModel(block)
    adapter = HuggingFaceModelAdapter.__new__(HuggingFaceModelAdapter)
    adapter._hf_model = hf_model
    adapter._model = FakeLensModel(block)
    adapter.tokenizer = FakeTokenizer()
    return adapter, hf_model


def test_intervention_changes_only_final_prefill_token_and_only_once() -> None:
    adapter, hf_model = make_generation_adapter()
    generated = adapter.generate_from_prompt(
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=1,
        layer=0,
        reconstructed_jspace=torch.tensor([0.5, 1.0, 1.5]),
        alpha=1.0,
    )
    assert generated.tolist() == [9]
    assert hf_model.prefill is not None and hf_model.decode is not None
    torch.testing.assert_close(hf_model.prefill[0, 0], torch.ones(3))
    torch.testing.assert_close(hf_model.prefill[0, -1], torch.tensor([0.5, 0.0, -0.5]))
    torch.testing.assert_close(hf_model.decode[0, 0], torch.ones(3))


def test_zero_strength_hook_matches_no_hook_generation_exactly() -> None:
    adapter, _ = make_generation_adapter()
    prompt = torch.tensor([[1, 2, 3]])
    hooked = adapter.generate_from_prompt(
        prompt,
        max_new_tokens=1,
        layer=0,
        reconstructed_jspace=torch.ones(3),
        alpha=0.0,
    )
    plain = adapter.generate_from_prompt(prompt, max_new_tokens=1)
    assert torch.equal(hooked, plain)


def test_generation_capture_records_only_the_prefill_final_token() -> None:
    adapter, hf_model = make_generation_adapter()
    generated, captured = adapter.generate_with_capture(
        torch.tensor([[1, 2, 3]]), max_new_tokens=1, layer=0
    )
    assert generated.tolist() == [9]
    torch.testing.assert_close(captured, torch.ones(3))
    assert hf_model.decode is not None
    torch.testing.assert_close(hf_model.decode[0, 0], torch.ones(3))


def test_intervention_rejects_wrong_reconstruction_shape() -> None:
    adapter, _ = make_generation_adapter()
    with pytest.raises(ValueError, match="shape does not match"):
        adapter.generate_from_prompt(
            torch.tensor([[1, 2, 3]]),
            max_new_tokens=1,
            layer=0,
            reconstructed_jspace=torch.ones(2),
            alpha=1.0,
        )
