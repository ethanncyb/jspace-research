"""Shared typed records used across the GSM8K J-Space pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Condition = Literal["baseline", "no_op", "intervention"]
TestType = Literal["full_answer", "gold_next_token"]
BackendName = Literal["auto", "mlx", "mps", "cuda", "rocm", "cpu"]
LayerMode = Literal["late", "all_fitted", "explicit", "range"]
TokenMode = Literal[
    "prompt_last",
    "generated_last",
    "all_generated",
    "generated_stride",
    "word_end",
    "explicit",
    "full_sequence",
]


@dataclass(frozen=True)
class GSM8KExample:
    example_id: str
    source_index: int
    question: str
    gold_rationale: str
    gold_answer: str
    question_sha256: str
    dataset: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "test"


@dataclass
class GenerationResult:
    generated_text: str
    generated_token_ids: list[int]
    n_prompt_tokens: int
    finish_reason: str
    elapsed_seconds: float
    prompt: str
    extra: dict[str, Any] = field(default_factory=dict)
