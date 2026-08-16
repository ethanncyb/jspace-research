from types import SimpleNamespace

import pytest
import torch
from torch import nn

from intervention import (
    HardStopSignal,
    InterventionConfig,
    PrefillIntervention,
    guarded_generate,
    project_attack_direction,
)
from model_hooks import ResidualHookController, validate_layer_layout
from probe import LayerProbeDetector
from tests.tiny import TinyDecoder


def test_hook_cleanup_and_baseline_delta() -> None:
    model = TinyDecoder(n_layers=4)
    ids_clean = model.encode("a")
    ids_full = model.encode("ab")
    controller = ResidualHookController(model.layers, [1, 3])
    before = [len(block._forward_hooks) for block in model.layers]
    with controller:
        controller.begin_clean()
        model(ids_clean)
        controller.begin_appended(
            ids_clean.shape[-1], prefill_length=ids_full.shape[-1]
        )
        model(ids_full)
    assert [len(block._forward_hooks) for block in model.layers] == before
    assert set(controller.deltas) == {1, 3}
    with torch.no_grad():
        hidden_clean = model.embed_tokens(ids_clean)
        hidden_full = model.embed_tokens(ids_full)
        for index, block in enumerate(model.layers):
            hidden_clean = block(hidden_clean)
            hidden_full = block(hidden_full)
            if index in {1, 3}:
                expected = hidden_full[:, -1].float() - hidden_clean[:, -1].float()
                assert torch.allclose(controller.deltas[index], expected)


class _TupleBlock(nn.Module):
    def forward(self, hidden: torch.Tensor):
        return hidden * 2, "cache"


def test_tensor_and_tuple_replacement_preserve_shape_and_members() -> None:
    block = _TupleBlock()

    def processor(_layer, hidden, _delta, _prefix):
        return hidden + 3, {"changed": True}

    controller = ResidualHookController([block], [0], processor=processor)
    with controller:
        controller.begin_clean()
        block(torch.ones(1, 1, 2))
        controller.begin_appended(1, prefill_length=2)
        output = block(torch.ones(1, 2, 2))
    assert output[1] == "cache"
    assert torch.equal(output[0], torch.full((1, 2, 2), 5.0))


def test_observation_only_layer_is_read_only() -> None:
    blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
    called = []

    def processor(layer, hidden, _delta, _prefix):
        called.append(layer)
        changed = hidden.clone()
        changed[:, 1:] += 1
        return changed, None

    controller = ResidualHookController(
        blocks, [1], observation_layers=[0, 1], processor=processor
    )
    clean = torch.zeros(1, 1, 2)
    full = torch.zeros(1, 2, 2)
    with controller:
        controller.begin_clean()
        for block in blocks:
            clean = block(clean)
        controller.begin_appended(1, prefill_length=2)
        for block in blocks:
            full = block(full)
    assert called == [1]
    assert torch.equal(full[:, 0], torch.zeros(1, 2))
    assert torch.equal(full[:, 1], torch.ones(1, 2))


def test_layer_type_validation() -> None:
    model = TinyDecoder(n_layers=4)
    text = SimpleNamespace(
        num_hidden_layers=4,
        hidden_size=8,
        layer_types=["gdn", "full_attention", "gdn", "full_attention"],
    )
    model.config = SimpleNamespace(get_text_config=lambda: text)
    assert validate_layer_layout(model, [1, 3]) == (1, 3)
    with pytest.raises(ValueError, match="not full-attention"):
        validate_layer_layout(model, [0])


def test_projection_formula_appended_only_and_beta_zero() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [2.0, 5.0], [7.0, 11.0]]])
    direction = torch.tensor([2.0, 0.0])
    changed = project_attack_direction(hidden, direction, prefix_length=1, beta=0.5)
    assert torch.equal(changed[:, :1], hidden[:, :1])
    assert torch.allclose(changed[:, 1:, 0], hidden[:, 1:, 0] * 0.5)
    assert torch.equal(changed[:, 1:, 1], hidden[:, 1:, 1])
    unchanged = project_attack_direction(hidden, direction, prefix_length=1, beta=0)
    assert unchanged is hidden


def _triggering_detector() -> LayerProbeDetector:
    detector = LayerProbeDetector([0], 2)
    with torch.no_grad():
        detector.heads["layer_0"].weight.zero_()
        detector.heads["layer_0"].bias.fill_(10)
    return detector


def test_hard_stop_signal_contains_diagnostic() -> None:
    processor = PrefillIntervention(
        _triggering_detector(), InterventionConfig(mode="hard_stop", threshold=0.5)
    )
    with pytest.raises(HardStopSignal) as caught:
        processor(0, torch.ones(1, 2, 2), torch.ones(1, 2), 1)
    assert caught.value.diagnostic.layer == 0


class _GeneratingTiny(TinyDecoder):
    def forward(self, input_ids: torch.Tensor, **_kwargs):
        return super().forward(input_ids)

    def generate(self, input_ids: torch.Tensor, **_kwargs):
        self(input_ids=input_ids)
        return torch.cat((input_ids, torch.ones((1, 1), dtype=torch.long)), dim=1)


def test_guarded_generation_catches_hard_stop_and_discards_output() -> None:
    model = _GeneratingTiny(n_layers=1, d_model=2)
    result = guarded_generate(
        model,
        model.tokenizer,
        "clean",
        " appended",
        _triggering_detector(),
        InterventionConfig(mode="hard_stop", threshold=0.5, refusal_text="refused"),
        max_length=64,
    )
    assert result.stopped
    assert result.text == "refused"
    assert result.token_ids is None


def test_later_checkpoint_sees_corrected_stream() -> None:
    detector = LayerProbeDetector([0, 1], 2)
    with torch.no_grad():
        for layer in (0, 1):
            head = detector.heads[f"layer_{layer}"]
            head.weight.copy_(torch.tensor([[1.0, 0.0]]))
            head.bias.fill_(10)
    processor = PrefillIntervention(
        detector, InterventionConfig(mode="circuit_breaker", threshold=0.5, beta=1)
    )
    blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
    controller = ResidualHookController(blocks, [0, 1], processor=processor)
    clean = torch.zeros(1, 1, 2)
    full = torch.ones(1, 2, 2)
    with controller:
        controller.begin_clean()
        for block in blocks:
            clean = block(clean)
        controller.begin_appended(1, prefill_length=2)
        for block in blocks:
            full = block(full)
    assert controller.deltas[0][0, 0] == 1
    assert torch.allclose(controller.deltas[1][0, 0], torch.tensor(0.0), atol=1e-6)
