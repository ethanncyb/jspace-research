"""Thresholded hard-stop and residual drift-direction interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from promptguard.config import InterventionConfig
from promptguard.drift_probe import DriftProbe, extract_appended_delta


@dataclass
class InterventionResult:
    text: str
    drift_score: float
    layer_scores: dict[int, float] = field(default_factory=dict)
    triggered: bool = False
    mode: str = "none"


class InterventionController:
    """Make one pre-generation decision and, if needed, build a WRITE hook."""

    VALID_MODES = {"circuit_breaker", "hard_stop"}

    def __init__(self, probe: DriftProbe, config: InterventionConfig) -> None:
        if config.mode not in self.VALID_MODES:
            raise ValueError(f"unsupported intervention mode: {config.mode}")
        if not 0 <= config.threshold <= 1:
            raise ValueError("intervention threshold must be in [0, 1]")
        if config.beta < 0:
            raise ValueError("circuit-breaker beta must be non-negative")
        if config.positions not in {"all", "last_token"}:
            raise ValueError("intervention positions must be 'all' or 'last_token'")
        self.probe = probe
        self.config = config

    def triggered(self, score: float) -> bool:
        return score >= self.config.threshold

    def writer(self):
        beta = self.config.beta
        positions = self.config.positions

        def project(layer: int, hidden: torch.Tensor) -> torch.Tensor:
            direction = self.probe.drift_direction(layer).to(
                device=hidden.device, dtype=hidden.dtype
            )
            changed = hidden.clone()
            target = changed if positions == "all" else changed[:, -1:, :]
            coefficient = torch.einsum("...h,h->...", target, direction)
            target -= beta * coefficient.unsqueeze(-1) * direction
            return changed

        return project

    def run(
        self,
        hooked_model: Any,
        *,
        baseline_text: str,
        prompt: str,
        pooling: str = "last_token",
        max_new_tokens: int = 64,
        **generation_kwargs: Any,
    ) -> InterventionResult:
        deltas = extract_appended_delta(
            hooked_model, baseline_text, prompt, pooling=pooling
        )
        score_tensor, layer_tensors = self.probe.score(deltas)
        score = float(score_tensor.reshape(-1)[0])
        layer_scores = {
            layer: float(value.reshape(-1)[0]) for layer, value in layer_tensors.items()
        }
        if not self.triggered(score):
            text = hooked_model.generate(
                prompt, max_new_tokens=max_new_tokens, **generation_kwargs
            )
            return InterventionResult(text, score, layer_scores)
        if self.config.mode == "hard_stop":
            return InterventionResult(
                self.config.refusal,
                score,
                layer_scores,
                triggered=True,
                mode="hard_stop",
            )
        text = hooked_model.generate(
            prompt,
            writer=self.writer(),
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )
        return InterventionResult(
            text,
            score,
            layer_scores,
            triggered=True,
            mode="circuit_breaker",
        )
