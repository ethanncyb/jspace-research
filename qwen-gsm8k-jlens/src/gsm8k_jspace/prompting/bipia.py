"""BIPIA prompt templates (concatenated user prompt, official no-system path)."""

from __future__ import annotations

from gsm8k_jspace.config import AppConfig
from gsm8k_jspace.datasets.bipia import TASK_PROMPTS
from gsm8k_jspace.types import GSM8KExample


def render_prompt(example: GSM8KExample, cfg: AppConfig) -> str:
    del cfg
    task = str(example.extra.get("task") or "email")
    template = TASK_PROMPTS.get(task)
    if template is None:
        raise ValueError(f"unknown BIPIA task {task!r}")
    return template.format(
        context=example.extra.get("context", ""),
        question=example.extra.get("question", example.question),
        code=example.extra.get("code", ""),
        error=example.extra.get("error", example.question),
    )
