"""Prefill-only circuit breaking and hard-stop behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from model_hooks import (
    ResidualHookController,
    encode_clean_and_appended,
    resolve_residual_stack,
)
from probe import LayerProbeDetector

InterventionMode = Literal["disabled", "circuit_breaker", "hard_stop"]


@dataclass(frozen=True)
class InterventionConfig:
    mode: InterventionMode = "circuit_breaker"
    threshold: float | None = None
    beta: float = 1.0
    refusal_text: str = "I cannot process that request."

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "circuit_breaker", "hard_stop"}:
            raise ValueError(f"unsupported intervention mode: {self.mode}")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.threshold is not None and not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")


@dataclass(frozen=True)
class TriggerDiagnostic:
    layer: int
    probability: float
    threshold: float
    mode: str
    beta: float


class HardStopSignal(RuntimeError):
    """Internal hook exception caught only by guarded generation wrappers."""

    def __init__(self, diagnostic: TriggerDiagnostic) -> None:
        super().__init__(f"prompt detector hard-stopped at layer {diagnostic.layer}")
        self.diagnostic = diagnostic


@dataclass
class GuardedResult:
    text: str
    token_ids: torch.Tensor | None
    stopped: bool
    triggers: list[TriggerDiagnostic] = field(default_factory=list)
    layer_diagnostics: dict[int, dict[str, Any]] = field(default_factory=dict)


def project_attack_direction(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    *,
    prefix_length: int,
    beta: float,
) -> torch.Tensor:
    """Project a direction from appended positions, preserving the prefix."""

    if beta == 0:
        return hidden
    if not 0 <= prefix_length <= hidden.shape[1]:
        raise ValueError("prefix length is outside the hidden sequence")
    unit = direction.to(hidden.device, hidden.dtype)
    norm = torch.linalg.vector_norm(unit.float())
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("attack direction must have a finite non-zero norm")
    unit = unit / norm.to(unit.dtype)
    changed = hidden.clone()
    appended = changed[:, prefix_length:, :]
    coefficients = torch.einsum("bsh,h->bs", appended, unit)
    appended.sub_(beta * coefficients.unsqueeze(-1) * unit)
    return changed


class PrefillIntervention:
    """Callable hook processor that scores and immediately intervenes."""

    def __init__(
        self, detector: LayerProbeDetector, config: InterventionConfig
    ) -> None:
        self.detector = detector
        self.config = config
        self.triggers: list[TriggerDiagnostic] = []

    def reset(self) -> None:
        self.triggers.clear()

    def __call__(
        self,
        layer: int,
        hidden: torch.Tensor,
        delta: torch.Tensor,
        prefix_length: int,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        with torch.no_grad():
            probability = float(self.detector.probability(layer, delta).max().item())
        threshold = (
            self.detector.threshold(layer)
            if self.config.threshold is None
            else self.config.threshold
        )
        metadata: dict[str, Any] = {
            "probe_probability": probability,
            "threshold": threshold,
            "triggered": probability > threshold,
            "intervention": "none",
        }
        if probability <= threshold or self.config.mode == "disabled":
            return None, metadata

        diagnostic = TriggerDiagnostic(
            layer, probability, threshold, self.config.mode, self.config.beta
        )
        self.triggers.append(diagnostic)
        if self.config.mode == "hard_stop":
            raise HardStopSignal(diagnostic)

        metadata["intervention"] = "projection"
        if self.config.beta == 0:
            return None, metadata
        direction = self.detector.raw_attack_direction(layer)
        return (
            project_attack_direction(
                hidden,
                direction,
                prefix_length=prefix_length,
                beta=self.config.beta,
            ),
            metadata,
        )


def _input_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def guarded_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    clean_prompt: str,
    appended_text: str,
    detector: LayerProbeDetector,
    config: InterventionConfig,
    *,
    max_length: int = 2048,
    generation_kwargs: dict[str, Any] | None = None,
) -> GuardedResult:
    """Capture a baseline, then guard only the initial generation prefill."""

    input_device = _input_device(model)
    detector.to(input_device)
    pair = encode_clean_and_appended(
        tokenizer,
        clean_prompt,
        appended_text,
        max_length=max_length,
        device=input_device,
    )
    processor = PrefillIntervention(detector, config)
    controller = ResidualHookController(
        resolve_residual_stack(model),
        detector.layers,
        processor=processor,
    )
    kwargs = {
        "do_sample": False,
        "max_new_tokens": 64,
        "use_cache": True,
        **(generation_kwargs or {}),
    }
    try:
        with controller:
            controller.begin_clean()
            with torch.no_grad():
                model(input_ids=pair.clean_ids, use_cache=False)
            controller.begin_appended(
                pair.prefix_length, prefill_length=pair.full_ids.shape[-1]
            )
            with torch.no_grad():
                token_ids = model.generate(pair.full_ids, **kwargs)
            controller.disable()
    except HardStopSignal as signal:
        # No cache/output escapes the interrupted generate call.
        return GuardedResult(
            config.refusal_text,
            None,
            True,
            triggers=[signal.diagnostic],
            layer_diagnostics={
                layer: diagnostic.metadata
                for layer, diagnostic in controller.diagnostics.items()
            },
        )
    text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    return GuardedResult(
        text,
        token_ids.detach().cpu(),
        False,
        triggers=list(processor.triggers),
        layer_diagnostics={
            layer: diagnostic.metadata
            for layer, diagnostic in controller.diagnostics.items()
        },
    )
