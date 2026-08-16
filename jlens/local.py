# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Prompt-local Jacobian readout and causal residual-stream steering.

The corpus-fitted :class:`~jlens.lens.JacobianLens` estimates a full
``d_model x d_model`` transport matrix.  That object is useful, but fitting it
becomes prohibitively expensive for the larger Qwen3 checkpoints.  This module
provides a deliberately narrower quantity that is exact for one prompt and one
target token::

    g[l, p] = d logit(target) / d residual[l, p]

It is a sensitivity measurement, not a decoder for hidden thoughts.  The same
gradient can be used as a causal steering direction and compared with a
matched-norm random direction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from jlens.hooks import ActivationRecorder, ActivationSteerer
from jlens.protocol import LensModel

DEFAULT_LOCAL_STRENGTHS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4)
DEFAULT_DEPTH_FRACTIONS = (0.25, 0.5, 0.75)


def relative_layers(
    n_layers: int,
    fractions: Sequence[float] = DEFAULT_DEPTH_FRACTIONS,
) -> list[int]:
    """Map depth fractions to unique zero-based residual-block indices."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if not fractions:
        raise ValueError("fractions must not be empty")
    layers: list[int] = []
    for fraction in fractions:
        if not math.isfinite(float(fraction)) or not 0 <= fraction <= 1:
            raise ValueError("depth fractions must be finite and in [0, 1]")
        layer = round(float(fraction) * (n_layers - 1))
        if layer not in layers:
            layers.append(layer)
    return layers


def _input_ids(
    model: LensModel, prompt: str | torch.Tensor, max_seq_len: int
) -> torch.Tensor:
    if isinstance(prompt, str):
        value = model.encode(prompt, max_length=max_seq_len)
    elif torch.is_tensor(prompt):
        value = prompt
    else:
        raise TypeError("prompt must be text or an input_ids tensor")
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError("local Jacobian APIs require input_ids shape [1, seq_len]")
    if value.shape[1] == 0:
        raise ValueError("prompt must contain at least one token")
    return value


def _validate_layers(model: LensModel, layers: Sequence[int] | None) -> list[int]:
    resolved = list(range(model.n_layers)) if layers is None else list(layers)
    if not resolved:
        raise ValueError("layers must not be empty")
    if len(set(resolved)) != len(resolved):
        raise ValueError("layers must not contain duplicates")
    bad = sorted(layer for layer in resolved if not 0 <= layer < model.n_layers)
    if bad:
        raise ValueError(
            f"layers {bad} out of range for a {model.n_layers}-layer model"
        )
    return sorted(resolved)


@dataclass
class LocalJacobianResult:
    """Exact target-logit gradients for one teacher-forced prompt."""

    input_ids: torch.Tensor
    target_token_id: int
    layers: list[int]
    target_position: int
    gradients: dict[int, torch.Tensor]
    clean_activations: dict[int, torch.Tensor]
    clean_logits: torch.Tensor

    def sensitivity(self) -> torch.Tensor:
        """Return ``[n_layers, seq_len]`` L2 gradient norms."""
        return torch.stack(
            [self.gradients[layer][0].float().norm(dim=-1) for layer in self.layers]
        )


@dataclass(frozen=True)
class LocalSteeringResult:
    """Diagnostics from one single-layer local-Jacobian intervention."""

    layer: int
    position: int
    strength: float
    control: Literal["local_jacobian", "random"]
    target_token_id: int
    clean_rank: int
    steered_rank: int
    clean_top1: bool
    steered_top1: bool
    reciprocal_rank: float
    target_logit_lift: float
    target_logprob_lift: float
    kl_divergence: float
    clean_residual_norm: float
    delta_norm: float
    clean_top_token_id: int
    steered_top_token_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_local_jacobian(
    model: LensModel,
    prompt: str | torch.Tensor,
    *,
    target_token_id: int,
    layers: Sequence[int] | None = None,
    target_position: int = -1,
    max_seq_len: int = 512,
) -> LocalJacobianResult:
    """Compute ``d target_logit / d residual`` at all requested sites.

    Parameters are expected to be frozen (as they are in
    :class:`~jlens.hf.HFLensModel`).  Only activation gradients are requested,
    so this function does not populate parameter ``.grad`` fields.
    """
    input_ids = _input_ids(model, prompt, max_seq_len)
    layers = _validate_layers(model, layers)
    vocab_size = int(model.unembedding_weight.shape[0])
    if not 0 <= target_token_id < vocab_size:
        raise ValueError(
            f"target_token_id {target_token_id} out of range for vocabulary "
            f"size {vocab_size}"
        )
    position = target_position + input_ids.shape[1] if target_position < 0 else target_position
    if not 0 <= position < input_ids.shape[1]:
        raise ValueError(
            f"target_position {target_position} out of range for sequence length "
            f"{input_ids.shape[1]}"
        )

    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    with (
        ActivationRecorder(
            model.layers, at=record_at, start_graph_at=min(layers)
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids)
        final_residual = recorder.activations[final_layer][:, position].float()
        logits = model.unembed(final_residual).float()
        target_logit = logits[0, target_token_id]
        requested = [recorder.activations[layer] for layer in layers]
        gradients = torch.autograd.grad(
            target_logit,
            requested,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

    by_layer = {
        layer: gradient.detach().float().cpu()
        for layer, gradient in zip(layers, gradients, strict=True)
    }
    if any(
        not torch.isfinite(gradient).all() for gradient in by_layer.values()
    ):
        raise RuntimeError("local Jacobian contains non-finite values")
    return LocalJacobianResult(
        input_ids=input_ids.detach(),
        target_token_id=int(target_token_id),
        layers=layers,
        target_position=position,
        gradients=by_layer,
        clean_activations={
            layer: recorder.activations[layer].detach().cpu() for layer in record_at
        },
        clean_logits=logits.detach().cpu(),
    )


def _rank(logits: torch.Tensor, token_id: int) -> int:
    target = logits[token_id]
    return int((logits > target).sum().item())


@torch.no_grad()
def steer_local(
    model: LensModel,
    local: LocalJacobianResult,
    *,
    layer: int,
    position: int = -1,
    strength: float = 0.1,
    control: Literal["local_jacobian", "random"] = "local_jacobian",
    random_seed: int = 0,
) -> LocalSteeringResult:
    """Apply one matched single-layer intervention using a precomputed result."""
    if layer not in local.layers:
        raise ValueError(f"layer {layer} was not computed in the local Jacobian")
    if not math.isfinite(float(strength)) or strength < 0:
        raise ValueError("strength must be a finite non-negative number")
    if control not in ("local_jacobian", "random"):
        raise ValueError("control must be 'local_jacobian' or 'random'")
    input_ids = local.input_ids
    resolved_position = position + input_ids.shape[1] if position < 0 else position
    if not 0 <= resolved_position < input_ids.shape[1]:
        raise ValueError(
            f"position {position} out of range for sequence length {input_ids.shape[1]}"
        )

    clean_activation = local.clean_activations[layer][0, resolved_position].float()
    clean_norm = clean_activation.norm()
    if control == "local_jacobian":
        direction = local.gradients[layer][0, resolved_position].float()
    else:
        generator = torch.Generator(device="cpu").manual_seed(
            int(random_seed) + layer * 1_000_003
        )
        direction = torch.randn(model.d_model, generator=generator)
    direction_norm = direction.norm()
    if (
        not torch.isfinite(direction).all()
        or not torch.isfinite(direction_norm)
        or direction_norm <= 0
    ):
        raise ValueError("steering direction is zero or non-finite")
    delta = float(strength) * clean_norm * (direction / direction_norm)

    final_layer = model.n_layers - 1
    with ActivationSteerer(
        model.layers,
        deltas={layer: delta},
        positions=[resolved_position],
    ), ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(input_ids)
    steered_logits = model.unembed(
        recorder.activations[final_layer][0, local.target_position].float()
    ).float()
    clean_logits = local.clean_logits[0].float()
    target = local.target_token_id
    clean_rank = _rank(clean_logits, target)
    steered_rank = _rank(steered_logits, target)
    clean_logprobs = F.log_softmax(clean_logits, dim=-1)
    steered_logprobs = F.log_softmax(steered_logits, dim=-1)
    kl = F.kl_div(
        steered_logprobs,
        clean_logprobs.exp(),
        reduction="sum",
    )
    return LocalSteeringResult(
        layer=layer,
        position=resolved_position,
        strength=float(strength),
        control=control,
        target_token_id=target,
        clean_rank=clean_rank,
        steered_rank=steered_rank,
        clean_top1=clean_rank == 0,
        steered_top1=steered_rank == 0,
        reciprocal_rank=1.0 / (steered_rank + 1),
        target_logit_lift=float(steered_logits[target] - clean_logits[target]),
        target_logprob_lift=float(
            steered_logprobs[target] - clean_logprobs[target]
        ),
        kl_divergence=float(kl),
        clean_residual_norm=float(clean_norm),
        delta_norm=float(delta.norm()),
        clean_top_token_id=int(clean_logits.argmax()),
        steered_top_token_id=int(steered_logits.argmax()),
    )


def run_local_steering_sweep(
    model: LensModel,
    local: LocalJacobianResult,
    *,
    layers: Sequence[int] | None = None,
    strengths: Sequence[float] = DEFAULT_LOCAL_STRENGTHS,
    position: int = -1,
    random_seed: int = 0,
) -> list[LocalSteeringResult]:
    """Run baseline once, then local and random controls per layer/strength."""
    selected = local.layers if layers is None else list(layers)
    if not selected:
        raise ValueError("layers must not be empty")
    strengths = [float(value) for value in strengths]
    if not strengths or strengths[0] != 0 or strengths.count(0.0) != 1:
        raise ValueError("strengths must begin with exactly one zero baseline")
    baseline = steer_local(
        model,
        local,
        layer=selected[0],
        position=position,
        strength=0,
        control="local_jacobian",
        random_seed=random_seed,
    )
    rows = [baseline]
    for layer in selected:
        for strength in strengths[1:]:
            for control in ("local_jacobian", "random"):
                rows.append(
                    steer_local(
                        model,
                        local,
                        layer=layer,
                        position=position,
                        strength=strength,
                        control=control,
                        random_seed=random_seed,
                    )
                )
    return rows


def generate_local_comparison(
    hf_model: Any,
    model: LensModel,
    local: LocalJacobianResult,
    *,
    layer: int,
    strength: float = 0.4,
    position: int = -1,
    max_new_tokens: int = 32,
    random_seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Greedily compare clean/local/random continuations for a saved prefix.

    The intervention is active only during the full-prompt prefill. Cached
    one-token decoding steps are left untouched. This is intended for the small
    representative set rendered beside the heatmaps, not task-accuracy scoring.
    """
    if layer not in local.layers:
        raise ValueError(f"layer {layer} was not computed in the local Jacobian")
    input_ids = local.input_ids.to(model.input_device)
    prompt_length = input_ids.shape[1]
    resolved_position = position + prompt_length if position < 0 else position
    if not 0 <= resolved_position < prompt_length:
        raise ValueError(f"position {position} out of range for prompt")
    clean_norm = local.clean_activations[layer][0, resolved_position].float().norm()

    def direction(control: str) -> torch.Tensor:
        if control == "local_jacobian":
            value = local.gradients[layer][0, resolved_position].float()
        else:
            generator = torch.Generator(device="cpu").manual_seed(
                int(random_seed) + layer * 1_000_003
            )
            value = torch.randn(model.d_model, generator=generator)
        return value / value.norm()

    def generate(control: str | None) -> dict[str, Any]:
        handle = None
        if control is not None:
            delta = float(strength) * clean_norm * direction(control)

            def hook(module, inputs, output):
                hidden = output if torch.is_tensor(output) else output[0]
                if hidden.shape[1] != prompt_length:
                    return None
                changed = hidden.clone()
                changed[:, resolved_position] += delta.to(
                    hidden.device, hidden.dtype
                )
                if torch.is_tensor(output):
                    return changed
                return (changed, *output[1:])

            handle = model.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                output = hf_model.generate(
                    input_ids=input_ids,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=getattr(model.tokenizer, "pad_token_id", None)
                    or getattr(model.tokenizer, "eos_token_id", None),
                )
        finally:
            if handle is not None:
                handle.remove()
        generated = output[0, prompt_length:]
        eos = getattr(model.tokenizer, "eos_token_id", None)
        return {
            "text": model.tokenizer.decode(generated, skip_special_tokens=True),
            "generated_tokens": int(len(generated)),
            "truncated": bool(
                len(generated) >= max_new_tokens
                and (eos is None or int(generated[-1]) != int(eos))
            ),
        }

    return {
        "clean": generate(None),
        "local_jacobian": generate("local_jacobian"),
        "random": generate("random"),
    }
