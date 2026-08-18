# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""J-space direction construction and activation-steering tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from jlens.hooks import ActivationSteerer, GenerateActivationSteerer
from jlens.lens import JacobianLens

from .tiny import TinyDecoder


def _lens(model: TinyDecoder) -> JacobianLens:
    jacobians = {
        layer: torch.eye(model.d_model) * (layer + 1)
        for layer in range(model.n_layers - 1)
    }
    return JacobianLens(jacobians, n_prompts=1, d_model=model.d_model)


def test_direction_is_normalized_unembedding_times_jacobian():
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = _lens(model)
    token_id = 7
    layer = 1
    expected = model.unembedding_weight[token_id].float() @ lens.jacobians[layer]
    expected /= expected.norm()
    torch.testing.assert_close(lens.direction(model, layer, token_id), expected)


def test_direction_validates_layer_and_token():
    model = TinyDecoder()
    lens = _lens(model)
    with pytest.raises(ValueError, match="not in source_layers"):
        lens.direction(model, model.n_layers - 1, 0)
    with pytest.raises(ValueError, match="out of range"):
        lens.direction(model, 0, model.unembedding_weight.shape[0])


class _TupleBlock(nn.Module):
    def forward(self, hidden):
        return hidden, "cache", 17


def test_activation_steerer_preserves_tuple_and_only_changes_positions():
    block = _TupleBlock()
    hidden = torch.zeros(1, 4, 3)
    deltas = {0: torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    with ActivationSteerer([block], deltas=deltas, positions=[1, 3]):
        output = block(hidden)
    assert output[1:] == ("cache", 17)
    torch.testing.assert_close(output[0][0, 0], torch.zeros(3))
    torch.testing.assert_close(output[0][0, 1], deltas[0][0])
    torch.testing.assert_close(output[0][0, 2], torch.zeros(3))
    torch.testing.assert_close(output[0][0, 3], deltas[0][1])


def test_strength_zero_matches_clean_pass_and_reports_diagnostics():
    model = TinyDecoder()
    lens = _lens(model)
    result = lens.steer(
        model,
        "abcdef",
        target_token_id=3,
        layers=[1, 2],
        positions=[-1],
        strength=0,
    )
    torch.testing.assert_close(result.clean_logits, result.steered_logits)
    torch.testing.assert_close(result.target_logit_lift, torch.zeros(1))
    torch.testing.assert_close(result.kl_divergence, torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(result.clean_target_ranks, result.steered_target_ranks)


def test_steer_uses_clean_position_norm_and_random_control_is_deterministic():
    model = TinyDecoder()
    lens = _lens(model)
    kwargs = dict(
        target_token_id=4,
        layers=[1],
        positions=[-1],
        strength=0.25,
        direction_mode="random",
        random_seed=9,
    )
    first = lens.steer(model, "abcdef", **kwargs)
    second = lens.steer(model, "abcdef", **kwargs)
    torch.testing.assert_close(first.steered_logits, second.steered_logits)
    position = first.positions[0]
    observed = (
        first.steered_activations[1][0, position]
        - first.clean_activations[1][0, position]
    ).float()
    expected_norm = 0.25 * first.clean_norms[1][0]
    torch.testing.assert_close(observed.norm().cpu(), expected_norm, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layers": []}, "layers must not be empty"),
        ({"layers": [99]}, "out of range"),
        ({"positions": [99]}, "position 99 out of range"),
        ({"positions": []}, "positions must not be empty"),
    ],
)
def test_steer_rejects_invalid_sites(kwargs, message):
    model = TinyDecoder()
    lens = _lens(model)
    base = {"target_token_id": 2, "layers": [1], "positions": [-1]}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        lens.steer(model, "abc", **base)


class _IdentityBlock(nn.Module):
    def forward(self, hidden):
        return hidden


def test_generate_steerer_prompt_pass_only_last_prompt_token():
    block = _IdentityBlock()
    prompt_len = 4
    hidden = torch.zeros(1, prompt_len, 3)
    delta = torch.tensor([1.0, 2.0, 3.0])
    with GenerateActivationSteerer(
        [block], deltas={0: delta}, prompt_len=prompt_len, decode_mode="prompt_pass"
    ):
        output = block(hidden.clone())
    torch.testing.assert_close(output[0, :3], torch.zeros(3, 3))
    torch.testing.assert_close(output[0, 3], delta)


def test_generate_steerer_prompt_pass_leaves_incremental_unchanged():
    block = _IdentityBlock()
    hidden = torch.ones(1, 1, 3)
    delta = torch.tensor([1.0, 2.0, 3.0])
    with GenerateActivationSteerer(
        [block], deltas={0: delta}, prompt_len=4, decode_mode="prompt_pass"
    ):
        output = block(hidden)
    torch.testing.assert_close(output, hidden)


def test_generate_steerer_every_step_adds_on_incremental():
    block = _IdentityBlock()
    hidden = torch.zeros(1, 1, 3)
    delta = torch.tensor([1.0, 2.0, 3.0])
    with GenerateActivationSteerer(
        [block], deltas={0: delta}, prompt_len=4, decode_mode="every_step"
    ):
        output = block(hidden.clone())
    torch.testing.assert_close(output[0, -1], delta)


def test_generate_steerer_every_step_growing_sequence_from_last_prompt():
    block = _IdentityBlock()
    prompt_len = 4
    extra = 2
    hidden = torch.zeros(1, prompt_len + extra, 3)
    delta = torch.tensor([1.0, 2.0, 3.0])
    with GenerateActivationSteerer(
        [block], deltas={0: delta}, prompt_len=prompt_len, decode_mode="every_step"
    ):
        output = block(hidden.clone())
    torch.testing.assert_close(output[0, : prompt_len - 1], torch.zeros(prompt_len - 1, 3))
    for index in range(prompt_len - 1, prompt_len + extra):
        torch.testing.assert_close(output[0, index], delta)


def test_generate_steerer_prompt_pass_growing_sequence_only_last_prompt():
    block = _IdentityBlock()
    prompt_len = 4
    hidden = torch.zeros(1, prompt_len + 2, 3)
    delta = torch.tensor([1.0, 2.0, 3.0])
    with GenerateActivationSteerer(
        [block], deltas={0: delta}, prompt_len=prompt_len, decode_mode="prompt_pass"
    ):
        output = block(hidden.clone())
    expected = torch.zeros_like(hidden)
    expected[0, prompt_len - 1] = delta
    torch.testing.assert_close(output, expected)


def test_generate_steerer_accepts_row_vector_delta_and_preserves_tuple():
    block = _TupleBlock()
    hidden = torch.zeros(1, 4, 3)
    delta = torch.tensor([[1.0, 2.0, 3.0]])
    with GenerateActivationSteerer(
        [block], deltas={0: delta}, prompt_len=4, decode_mode="prompt_pass"
    ):
        output = block(hidden)
    assert output[1:] == ("cache", 17)
    torch.testing.assert_close(output[0][0, -1], delta[0])


def test_generate_steerer_rejects_invalid_mode_and_prompt_len():
    block = _IdentityBlock()
    with pytest.raises(ValueError, match="prompt_len"):
        GenerateActivationSteerer(
            [block], deltas={0: torch.ones(3)}, prompt_len=0
        )
    with pytest.raises(ValueError, match="decode_mode"):
        GenerateActivationSteerer(
            [block],
            deltas={0: torch.ones(3)},
            prompt_len=2,
            decode_mode="both",
        )


def test_steer_generate_runs_hooks_during_fake_generate():
    model = TinyDecoder()
    lens = _lens(model)
    ids = model.encode("abc")

    class FakeHF(nn.Module):
        def generate(self, input_ids, **kwargs):
            model.forward(input_ids)
            model.forward(input_ids[:, -1:])
            return torch.cat([input_ids, input_ids[:, -1:]], dim=1)

    out = lens.steer_generate(
        FakeHF(),
        model,
        ids,
        target_token_id=3,
        strength=0.25,
        decode_mode="every_step",
        layers=[1],
        max_new_tokens=1,
    )
    assert out.shape == (1, ids.shape[1] + 1)


def test_steer_generate_rejects_invalid_decode_mode():
    model = TinyDecoder()
    lens = _lens(model)
    with pytest.raises(ValueError, match="decode_mode"):
        lens.steer_generate(
            nn.Linear(1, 1),
            model,
            "abc",
            target_token_id=2,
            layers=[1],
            decode_mode="both",
        )
