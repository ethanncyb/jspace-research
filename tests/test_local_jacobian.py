# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from jlens.local import (
    compute_local_jacobian,
    generate_local_comparison,
    relative_layers,
    run_local_steering_sweep,
    steer_local,
)

from .tiny import TinyDecoder


def test_relative_layers_are_matched_by_fraction():
    assert relative_layers(4) == [1, 2]
    assert relative_layers(64) == [16, 32, 47]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        relative_layers(4, [-0.1])


def test_local_jacobian_matches_finite_difference_direction():
    model = TinyDecoder(seed=4)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target = 7
    local = compute_local_jacobian(
        model, "abcdef", target_token_id=target, layers=[1]
    )
    gradient = local.gradients[1][0, -1]
    direction = gradient / gradient.norm()

    def target_logit(scale: float) -> float:
        from jlens.hooks import ActivationRecorder, ActivationSteerer

        with ActivationSteerer(
            model.layers, deltas={1: scale * direction}, positions=[-1]
        ), ActivationRecorder(model.layers, at=[model.n_layers - 1]) as recorder:
            model.forward(local.input_ids)
        return float(model.unembed(recorder.activations[model.n_layers - 1][0, -1])[target])

    epsilon = 1e-3
    observed = (target_logit(epsilon) - target_logit(-epsilon)) / (2 * epsilon)
    assert observed == pytest.approx(float(gradient.norm()), rel=2e-2, abs=2e-3)


def test_positive_local_intervention_increases_target_for_small_strength():
    model = TinyDecoder(seed=8)
    local = compute_local_jacobian(
        model, "steer me", target_token_id=5, layers=[2]
    )
    result = steer_local(model, local, layer=2, strength=1e-3)
    assert result.target_logit_lift > 0
    assert result.delta_norm == pytest.approx(
        1e-3 * result.clean_residual_norm, rel=1e-5
    )


def test_random_control_is_deterministic_and_sweep_shape_is_fixed():
    model = TinyDecoder(seed=9)
    local = compute_local_jacobian(
        model, "abcdef", target_token_id=4, layers=[1, 2]
    )
    first = steer_local(
        model, local, layer=1, strength=0.1, control="random", random_seed=17
    )
    second = steer_local(
        model, local, layer=1, strength=0.1, control="random", random_seed=17
    )
    assert first == second
    rows = run_local_steering_sweep(
        model, local, strengths=[0, 0.1], random_seed=17
    )
    assert [(row.layer, row.strength, row.control) for row in rows] == [
        (1, 0.0, "local_jacobian"),
        (1, 0.1, "local_jacobian"),
        (1, 0.1, "random"),
        (2, 0.1, "local_jacobian"),
        (2, 0.1, "random"),
    ]


def test_local_jacobian_does_not_populate_parameter_gradients():
    model = TinyDecoder()
    compute_local_jacobian(model, "abcdef", target_token_id=2, layers=[0, 2])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_generation_preview_cleans_up_prefill_hook():
    class GeneratingTiny(TinyDecoder):
        def generate(self, input_ids, **kwargs):
            self.forward(input_ids)
            return torch.cat((input_ids, input_ids.new_tensor([[2, 3]])), dim=1)

    model = GeneratingTiny()
    local = compute_local_jacobian(
        model, "abc", target_token_id=5, layers=[1]
    )
    before = len(model.layers[1]._forward_hooks)
    comparison = generate_local_comparison(
        model, model, local, layer=1, max_new_tokens=2
    )
    assert set(comparison) == {"clean", "local_jacobian", "random"}
    assert len(model.layers[1]._forward_hooks) == before
