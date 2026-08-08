# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens: fit and apply the average input-output Jacobian as a readout
of decoder-transformer residuals."""

from jlens._logging import configure_logging
from jlens.benchmark import (
    BenchmarkObservation,
    GoldNextTokenCase,
    load_gsm8k_cases,
    load_humaneval_cases,
    run_gold_next_token_benchmark,
    summarize_benchmark,
)
from jlens.fitting import fit, jacobian_for_prompt
from jlens.guard_embed import cosine, embed_text, normalize_prompt
from jlens.guard_memory import GuardMemory, Incident
from jlens.guard_probe import SecurityProbe, Verdict
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder, ActivationSteerer
from jlens.lens import JacobianLens, SteeringResult
from jlens.protocol import LensModel
from jlens.vis import (
    SteeringComparisonData,
    build_steering_comparison_page,
    compute_steering_comparison,
)

__all__ = [
    "ActivationRecorder",
    "ActivationSteerer",
    "BenchmarkObservation",
    "GoldNextTokenCase",
    "GuardMemory",
    "HFLensModel",
    "Incident",
    "JacobianLens",
    "SecurityProbe",
    "SteeringResult",
    "SteeringComparisonData",
    "Layout",
    "LensModel",
    "Verdict",
    "configure_logging",
    "build_steering_comparison_page",
    "compute_steering_comparison",
    "cosine",
    "embed_text",
    "fit",
    "from_hf",
    "jacobian_for_prompt",
    "load_gsm8k_cases",
    "load_humaneval_cases",
    "normalize_prompt",
    "run_gold_next_token_benchmark",
    "summarize_benchmark",
]
