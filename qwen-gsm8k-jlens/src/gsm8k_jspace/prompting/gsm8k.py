"""Versioned GSM8K prompt templates."""

from __future__ import annotations

from gsm8k_jspace.config import AppConfig
from gsm8k_jspace.types import GSM8KExample

ZERO_SHOT_COT_V1 = (
    "Solve the following grade-school math problem. Show the reasoning, then "
    "put the final numeric answer after ####.\n\n"
    "Question: {question}\n"
    "Answer:"
)

DEFAULT_FEW_SHOT: list[tuple[str, str]] = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell altogether "
        "in April and May?",
        "Natalia sold 48/2 = 24 clips in May.\n"
        "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
        "#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
        "minutes of babysitting. How much did she earn?",
        "Weng earns 12/60 = $0.2 per minute.\n"
        "Working 50 minutes, she earned 0.2 x 50 = $10.\n"
        "#### 10",
    ),
]


def render_prompt(example: GSM8KExample, cfg: AppConfig) -> str:
    template = cfg.prompt.template
    n_shot = int(cfg.prompt.few_shot_examples)
    if template == "zero_shot_cot_v1" and n_shot == 0:
        return ZERO_SHOT_COT_V1.format(question=example.question)
    if template in {"zero_shot_cot_v1", "few_shot_cot_v1"}:
        shots = DEFAULT_FEW_SHOT[:n_shot]
        parts = [
            "Solve each grade-school math problem. Show the reasoning, then "
            "put the final numeric answer after ####.",
            "",
        ]
        for question, answer in shots:
            parts.append(f"Question: {question}")
            parts.append(f"Answer: {answer}")
            parts.append("")
        parts.append(f"Question: {example.question}")
        parts.append("Answer:")
        return "\n".join(parts)
    raise ValueError(f"unknown prompt template {template!r}")
