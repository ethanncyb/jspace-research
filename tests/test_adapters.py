from __future__ import annotations

from types import SimpleNamespace

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


def test_model_placement_falls_back_to_parameter_devices() -> None:
    class Parameter:
        def __init__(self, device: str) -> None:
            self.device = torch.device(device)

    class Model:
        def __init__(self, devices: tuple[str, ...]) -> None:
            self._parameters = [Parameter(device) for device in devices]

        def parameters(self) -> list[Parameter]:
            return self._parameters

    _validate_primary_gpu_placement(Model(("cuda:0", "cuda:0")))
    with pytest.raises(RuntimeError, match="fit entirely on CUDA GPU 0"):
        _validate_primary_gpu_placement(Model(("cuda:0", "cpu")))


def test_model_context_length_comes_from_the_pinned_text_config() -> None:
    adapter = HuggingFaceModelAdapter.__new__(HuggingFaceModelAdapter)
    adapter._hf_model = SimpleNamespace(
        config=SimpleNamespace(
            max_position_embeddings=1024,
            text_config=SimpleNamespace(max_position_embeddings=262144),
        )
    )
    assert adapter.context_length == 262144


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


def test_intervention_leaves_prefill_untouched_and_edits_decode() -> None:
    adapter, hf_model = make_generation_adapter()
    generated = adapter.generate_from_prompt(
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=1,
        layer=0,
        dictionary=torch.eye(3),
        sparsity_k=3,
        screen_candidates=3,
        alpha=1.0,
    )
    assert generated.tolist() == [9]
    assert hf_model.prefill is not None and hf_model.decode is not None
    torch.testing.assert_close(hf_model.prefill[0, 0], torch.ones(3))
    torch.testing.assert_close(hf_model.prefill[0, -1], torch.ones(3))
    torch.testing.assert_close(hf_model.decode[0, 0], torch.zeros(3))


def test_zero_strength_hook_matches_no_hook_generation_exactly() -> None:
    adapter, _ = make_generation_adapter()
    prompt = torch.tensor([[1, 2, 3]])
    hooked = adapter.generate_from_prompt(
        prompt,
        max_new_tokens=1,
        layer=0,
        dictionary=torch.eye(3),
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
            dictionary=torch.eye(2),
            alpha=1.0,
        )


@pytest.mark.parametrize("window", [1, 5, 10, 15])
def test_output_window_reconstructs_each_current_token_and_stops(monkeypatch, window):
    from jspace_research.phase1 import jspace

    adapter, hf_model = make_generation_adapter()
    seen = []
    outputs = []

    def reconstruct(hidden, dictionary, **kwargs):
        seen.append(hidden.clone())
        return hidden * 0.5, torch.zeros((1, 1)), torch.zeros((1, 1))

    def generation(*, input_ids, **kwargs):
        # Single-token prompts must still be recognized as prefill.
        outputs.append(hf_model.block(torch.full((1, 1, 3), 20.0))[0])
        for step in range(window + 2):
            outputs.append(hf_model.block(torch.full((1, 1, 3), float(step + 1)))[0])
        return torch.cat([input_ids, torch.full((1, window + 3), 9)], dim=-1)

    monkeypatch.setattr(jspace, "screened_nonnegative_pursuit", reconstruct)
    monkeypatch.setattr(hf_model, "generate", generation)
    stats = {}
    adapter.generate_from_prompt(
        torch.tensor([[1]]),
        max_new_tokens=window + 3,
        layer=0,
        dictionary=torch.eye(3),
        output_window=window,
        alpha=0.5,
        intervention_stats=stats,
    )
    assert len(seen) == window
    torch.testing.assert_close(outputs[0], torch.full((1, 1, 3), 21.0))
    for step in range(window):
        torch.testing.assert_close(seen[step], torch.full((1, 3), float(step + 2)))
        torch.testing.assert_close(outputs[step + 1], torch.full((1, 1, 3), (step + 2) * 0.75))
    torch.testing.assert_close(outputs[window + 1], torch.full((1, 1, 3), float(window + 2)))
    assert stats == {"processed_output_tokens": window, "edited_output_tokens": window}
    assert not hf_model.block._forward_hooks


@pytest.mark.parametrize("generated_count", [1, 3])
def test_early_eos_allows_shorter_window(monkeypatch, generated_count):
    adapter, hf_model = make_generation_adapter()

    def generation(*, input_ids, **kwargs):
        hf_model.block(torch.ones((1, input_ids.shape[1], 3)))
        for _ in range(generated_count - 1):
            hf_model.block(torch.ones((1, 1, 3)))
        return torch.cat([input_ids, torch.full((1, generated_count), 2)], dim=-1)

    monkeypatch.setattr(hf_model, "generate", generation)
    stats = {}
    adapter.generate_from_prompt(
        torch.tensor([[1, 2]]),
        max_new_tokens=30,
        layer=0,
        dictionary=torch.eye(3),
        output_window=15,
        alpha=1.0,
        intervention_stats=stats,
    )
    assert stats["edited_output_tokens"] == generated_count - 1


def test_intervention_hook_removed_on_reconstruction_error(monkeypatch):
    from jspace_research.phase1 import jspace

    adapter, hf_model = make_generation_adapter()

    def fail(*args, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(jspace, "screened_nonnegative_pursuit", fail)
    with pytest.raises(RuntimeError, match="solver failed"):
        adapter.generate_from_prompt(
            torch.tensor([[1, 2]]), max_new_tokens=3, layer=0, dictionary=torch.eye(3), alpha=1.0
        )
    assert not hf_model.block._forward_hooks


def test_real_cached_generation_keeps_first_token_and_zero_strength_equivalence():
    from types import SimpleNamespace

    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(42)
    hf_model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=20,
            n_positions=32,
            n_embd=8,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
    ).eval()
    adapter = HuggingFaceModelAdapter.__new__(HuggingFaceModelAdapter)
    adapter._hf_model = hf_model
    adapter._model = SimpleNamespace(
        layers=hf_model.transformer.h, d_model=8, n_layers=2, input_device="cpu"
    )
    adapter.tokenizer = FakeTokenizer()
    prompt = torch.tensor([[1, 4, 5]])
    plain = adapter.generate_from_prompt(prompt, max_new_tokens=6)
    stats = {}
    zero = adapter.generate_from_prompt(
        prompt,
        max_new_tokens=6,
        layer=0,
        dictionary=torch.eye(8),
        alpha=0.0,
        output_window=5,
        intervention_stats=stats,
    )
    assert torch.equal(plain, zero)
    assert stats == {"processed_output_tokens": 5, "edited_output_tokens": 0}
    edited = adapter.generate_from_prompt(
        prompt, max_new_tokens=6, layer=0, dictionary=torch.eye(8), alpha=1.0, output_window=5
    )
    assert edited[0] == plain[0]
