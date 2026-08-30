"""GSM8K loading with stable IDs and deterministic subsets."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

from gsm8k_jspace.config import AppConfig
from gsm8k_jspace.datasets.selection import select_examples
from gsm8k_jspace.types import GSM8KExample

_CALCULATOR_ANNOTATION = re.compile(r"<<[^<>]*>>")


def clean_gsm8k_answer(answer: str) -> str:
    """Remove GSM8K ``<<expression=result>>`` calculator annotations."""
    return _CALCULATOR_ANNOTATION.sub("", answer)


def extract_gold_answer(answer: str) -> str:
    cleaned = clean_gsm8k_answer(answer).strip()
    if "####" in cleaned:
        return cleaned.rsplit("####", 1)[-1].strip()
    return cleaned


def example_id_for_index(source_index: int, split: str = "test") -> str:
    return f"gsm8k_{split}_{source_index:06d}"


def question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def records_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    dataset: str,
    dataset_config: str,
    split: str,
) -> list[GSM8KExample]:
    examples: list[GSM8KExample] = []
    for index, row in enumerate(rows):
        question = str(row["question"])
        rationale = str(row["answer"])
        examples.append(
            GSM8KExample(
                example_id=example_id_for_index(index, split),
                source_index=index,
                question=question,
                gold_rationale=clean_gsm8k_answer(rationale),
                gold_answer=extract_gold_answer(rationale),
                question_sha256=question_sha256(question),
                dataset=dataset,
                dataset_config=dataset_config,
                split=split,
            )
        )
    return examples


def load_gsm8k_examples(
    cfg: AppConfig,
    *,
    rows: Iterable[dict[str, Any]] | None = None,
) -> list[GSM8KExample]:
    bench = cfg.benchmark
    if rows is None:
        from datasets import load_dataset

        loaded = load_dataset(bench.dataset, bench.dataset_config, split=bench.split)
        rows = [dict(row) for row in loaded]
    examples = records_from_rows(
        rows,
        dataset=bench.dataset,
        dataset_config=bench.dataset_config,
        split=bench.split,
    )
    selected = select_examples(
        examples,
        full_run=bench.full_run,
        subset_size=bench.subset_size,
        selection=bench.selection,
        seed=bench.selection_seed,
    )
    print(
        f"[gsm8k] loaded {len(selected)} examples "
        f"(full_run={bench.full_run}, selection={bench.selection})"
    )
    return selected
