"""Deterministic subset selection shared by every benchmark."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from gsm8k_jspace.types import GSM8KExample

SELECTION_EXTRA_KEYS = (
    "task",
    "attack_name",
    "attack_str",
    "position",
    "ideal",
    "user_tool",
    "attacker_tools",
    "attacker_instruction",
    "attack_type",
    "suite",
    "user_task_id",
    "injection_task_id",
    "attacker_action",
    "markers",
    "setting",
)


def select_examples(
    examples: Sequence[GSM8KExample],
    *,
    full_run: bool,
    subset_size: int,
    selection: str,
    seed: int,
) -> list[GSM8KExample]:
    ordered = list(examples)
    if selection == "shuffled":
        rng = random.Random(seed)
        rng.shuffle(ordered)
    if full_run:
        return ordered
    return ordered[: int(subset_size)]


def selection_records(examples: Sequence[GSM8KExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        row: dict[str, Any] = {
            "example_id": example.example_id,
            "source_index": example.source_index,
            "dataset": example.dataset,
            "dataset_config": example.dataset_config,
            "split": example.split,
            "question_sha256": example.question_sha256,
            "gold_answer": example.gold_answer,
        }
        for key in SELECTION_EXTRA_KEYS:
            if key in example.extra:
                row[key] = example.extra[key]
        rows.append(row)
    return rows


def completion_extra_fields(example: GSM8KExample) -> dict[str, Any]:
    return {
        key: example.extra[key]
        for key in SELECTION_EXTRA_KEYS
        if key in example.extra
    }
