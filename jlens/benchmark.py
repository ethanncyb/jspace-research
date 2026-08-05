# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Gold next-token steering benchmarks for HumanEval and GSM8K.

These utilities measure controllability on teacher-forced reference prefixes.
They do not measure pass@1, mathematical accuracy, or an improvement in model
reasoning. Dataset downloads require the optional ``datasets`` dependency.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from jlens.lens import JacobianLens
from jlens.protocol import LensModel

DEFAULT_STRENGTHS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4)


@dataclass(frozen=True)
class GoldNextTokenCase:
    dataset: Literal["humaneval", "gsm8k"]
    case_id: str
    input_ids: tuple[int, ...]
    target_token_id: int
    target_text: str


@dataclass(frozen=True)
class BenchmarkObservation:
    dataset: str
    case_id: str
    strength: float
    control: Literal["jspace", "random"]
    target_token_id: int
    target_text: str
    clean_rank: int
    steered_rank: int
    clean_top1: bool
    steered_top1: bool
    reciprocal_rank: float
    target_logit_lift: float
    kl_divergence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _decoded_is_alphanumeric(tokenizer: Any, token_id: int) -> bool:
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    return any(character.isalnum() for character in text)


def _case_from_reference(
    *,
    tokenizer: Any,
    dataset: Literal["humaneval", "gsm8k"],
    case_id: str,
    prompt: str,
    completion: str,
    rng: random.Random,
    max_seq_len: int,
) -> GoldNextTokenCase | None:
    prompt_ids = _token_ids(tokenizer, prompt)
    full_ids = _token_ids(tokenizer, prompt + completion)
    common = 0
    for prompt_id, full_id in zip(prompt_ids, full_ids, strict=False):
        if prompt_id != full_id:
            break
        common += 1
    # Tokenization can merge across the prompt/completion boundary. Starting
    # up to two tokens before the nominal boundary safely includes that merge
    # while keeping sampled targets in the reference continuation.
    completion_start = max(1, min(common, len(prompt_ids)) - 2)
    completion_end = len(full_ids)
    if completion_end - completion_start < 3:
        return None
    low = completion_start + math.floor((completion_end - completion_start) * 0.1)
    high = completion_start + math.ceil((completion_end - completion_start) * 0.9)
    candidates = [
        index
        for index in range(low, min(high, completion_end))
        if 0 < index < max_seq_len
        and _decoded_is_alphanumeric(tokenizer, full_ids[index])
    ]
    if not candidates:
        return None
    target_index = rng.choice(candidates)
    target_id = full_ids[target_index]
    return GoldNextTokenCase(
        dataset=dataset,
        case_id=case_id,
        input_ids=tuple(full_ids[:target_index]),
        target_token_id=target_id,
        target_text=tokenizer.decode(
            [target_id], clean_up_tokenization_spaces=False
        ),
    )


def _collect_cases(
    records: Iterable[dict[str, Any]],
    *,
    tokenizer: Any,
    dataset: Literal["humaneval", "gsm8k"],
    n_examples: int,
    seed: int,
    max_seq_len: int,
    fields: Callable[[dict[str, Any], int], tuple[str, str, str]],
) -> list[GoldNextTokenCase]:
    if n_examples <= 0:
        return []
    rng = random.Random(seed)
    cases: list[GoldNextTokenCase] = []
    for index, record in enumerate(records):
        case_id, prompt, completion = fields(record, index)
        case = _case_from_reference(
            tokenizer=tokenizer,
            dataset=dataset,
            case_id=case_id,
            prompt=prompt,
            completion=completion,
            rng=rng,
            max_seq_len=max_seq_len,
        )
        if case is not None:
            cases.append(case)
        if len(cases) == n_examples:
            break
    if len(cases) < n_examples:
        raise RuntimeError(
            f"only found {len(cases)} usable {dataset} examples; requested {n_examples}"
        )
    return cases


def load_humaneval_cases(
    tokenizer: Any,
    *,
    n_examples: int = 32,
    seed: int = 0,
    max_seq_len: int = 512,
) -> list[GoldNextTokenCase]:
    """Load deterministic HumanEval reference-prefix cases."""
    from datasets import load_dataset

    records = load_dataset("openai/openai_humaneval", split="test")
    return _collect_cases(
        records,
        tokenizer=tokenizer,
        dataset="humaneval",
        n_examples=n_examples,
        seed=seed,
        max_seq_len=max_seq_len,
        fields=lambda row, index: (
            str(row.get("task_id", index)),
            str(row["prompt"]),
            str(row["canonical_solution"]),
        ),
    )


_CALCULATOR_ANNOTATION = re.compile(r"<<[^<>]*>>")


def clean_gsm8k_answer(answer: str) -> str:
    """Remove GSM8K's ``<<expression=result>>`` calculator annotations."""
    return _CALCULATOR_ANNOTATION.sub("", answer)


def load_gsm8k_cases(
    tokenizer: Any,
    *,
    n_examples: int = 32,
    seed: int = 0,
    max_seq_len: int = 512,
) -> list[GoldNextTokenCase]:
    """Load deterministic GSM8K reference-prefix cases."""
    from datasets import load_dataset

    records = load_dataset("openai/gsm8k", "main", split="test")
    return _collect_cases(
        records,
        tokenizer=tokenizer,
        dataset="gsm8k",
        n_examples=n_examples,
        seed=seed,
        max_seq_len=max_seq_len,
        fields=lambda row, index: (
            str(index),
            f"Question: {row['question']}\nAnswer:",
            " " + clean_gsm8k_answer(str(row["answer"])),
        ),
    )


def run_gold_next_token_benchmark(
    model: LensModel,
    lens: JacobianLens,
    cases: Sequence[GoldNextTokenCase],
    *,
    strengths: Sequence[float] = DEFAULT_STRENGTHS,
    layers: Sequence[int] | None = None,
    random_seed: int = 0,
    progress: Callable[[int, int, GoldNextTokenCase, float, str], None] | None = None,
) -> list[BenchmarkObservation]:
    """Run J-space steering and matched random-direction controls.

    Strength zero is evaluated once as the J-space baseline. Every positive
    strength is evaluated in both conditions using the same clean-norm scale.
    """
    conditions = [
        (float(strength), control)
        for strength in strengths
        for control in (("jspace",) if strength == 0 else ("jspace", "random"))
    ]
    total = len(cases) * len(conditions)
    observations: list[BenchmarkObservation] = []
    completed = 0
    input_device = model.unembedding_weight.device
    for case_index, case in enumerate(cases):
        input_ids = torch.tensor(
            [case.input_ids], dtype=torch.long, device=input_device
        )
        for strength, control in conditions:
            if progress is not None:
                progress(completed, total, case, strength, control)
            result = lens.steer(
                model,
                input_ids,
                target_token_id=case.target_token_id,
                layers=layers,
                positions=[-1],
                strength=strength,
                direction_mode=control,
                random_seed=random_seed + case_index * 10_007,
            )
            clean_rank = int(result.clean_target_ranks[0])
            steered_rank = int(result.steered_target_ranks[0])
            observations.append(
                BenchmarkObservation(
                    dataset=case.dataset,
                    case_id=case.case_id,
                    strength=strength,
                    control=control,
                    target_token_id=case.target_token_id,
                    target_text=case.target_text,
                    clean_rank=clean_rank,
                    steered_rank=steered_rank,
                    clean_top1=clean_rank == 0,
                    steered_top1=steered_rank == 0,
                    reciprocal_rank=1.0 / (steered_rank + 1),
                    target_logit_lift=float(result.target_logit_lift[0]),
                    kl_divergence=float(result.kl_divergence[0]),
                )
            )
            completed += 1
    return observations


def summarize_benchmark(
    observations: Sequence[BenchmarkObservation],
) -> list[dict[str, float | int | str]]:
    """Aggregate observations by dataset, condition, and strength."""
    groups: dict[tuple[str, str, float], list[BenchmarkObservation]] = {}
    for observation in observations:
        key = (observation.dataset, observation.control, observation.strength)
        groups.setdefault(key, []).append(observation)
    summary: list[dict[str, float | int | str]] = []
    for (dataset, control, strength), rows in sorted(groups.items()):
        summary.append(
            {
                "dataset": dataset,
                "control": control,
                "strength": strength,
                "n": len(rows),
                "top1_rate": statistics.fmean(row.steered_top1 for row in rows),
                "median_rank": float(statistics.median(row.steered_rank for row in rows)),
                "mean_reciprocal_rank": statistics.fmean(
                    row.reciprocal_rank for row in rows
                ),
                "mean_logit_lift": statistics.fmean(
                    row.target_logit_lift for row in rows
                ),
                "mean_kl_divergence": statistics.fmean(
                    row.kl_divergence for row in rows
                ),
            }
        )
    return summary
