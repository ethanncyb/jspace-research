# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from jlens.benchmark import (
    BenchmarkObservation,
    GoldNextTokenCase,
    _collect_cases,
    clean_gsm8k_answer,
    run_gold_next_token_benchmark,
    summarize_benchmark,
)
from jlens.lens import JacobianLens

from .tiny import TinyDecoder


class _Tokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[0] + [ord(character) for character in text])

    def decode(self, ids, **kwargs):
        return "".join(chr(token_id) for token_id in ids if token_id)


def test_collect_cases_is_deterministic_and_targets_reference_tokens():
    records = [
        {"id": str(index), "prompt": f"prompt {index}: ", "answer": "alpha beta gamma"}
        for index in range(5)
    ]
    kwargs = dict(
        tokenizer=_Tokenizer(),
        dataset="humaneval",
        n_examples=3,
        seed=4,
        max_seq_len=512,
        fields=lambda row, index: (row["id"], row["prompt"], row["answer"]),
    )
    first = _collect_cases(records, **kwargs)
    second = _collect_cases(records, **kwargs)
    assert first == second
    assert len(first) == 3
    assert all(case.target_text.isalnum() for case in first)


def test_clean_gsm8k_answer_removes_only_annotations():
    answer = "48/2 = <<48/2=24>>24.\n#### 24"
    assert clean_gsm8k_answer(answer) == "48/2 = 24.\n#### 24"


def test_summary_reports_requested_metrics():
    rows = [
        BenchmarkObservation(
            dataset="gsm8k",
            case_id=str(index),
            strength=0.1,
            control="jspace",
            target_token_id=1,
            target_text="x",
            clean_rank=10,
            steered_rank=rank,
            clean_top1=False,
            steered_top1=rank == 0,
            reciprocal_rank=1 / (rank + 1),
            target_logit_lift=float(index + 1),
            kl_divergence=0.01 * index,
        )
        for index, rank in enumerate([0, 3])
    ]
    summary = summarize_benchmark(rows)[0]
    assert summary["top1_rate"] == 0.5
    assert summary["median_rank"] == 1.5
    assert summary["mean_reciprocal_rank"] == 0.625
    assert summary["mean_logit_lift"] == 1.5


def test_benchmark_runs_baseline_and_matched_control():
    model = TinyDecoder()
    lens = JacobianLens(
        {layer: model.layers[layer].linear.weight.detach().clone() + torch.eye(8)
         for layer in range(3)},
        n_prompts=1,
        d_model=8,
    )
    case = GoldNextTokenCase(
        dataset="gsm8k",
        case_id="tiny",
        input_ids=(0, 2, 3, 4),
        target_token_id=5,
        target_text="e",
    )
    rows = run_gold_next_token_benchmark(
        model, lens, [case], strengths=[0, 0.1], layers=[1], random_seed=3
    )
    assert [(row.strength, row.control) for row in rows] == [
        (0.0, "jspace"),
        (0.1, "jspace"),
        (0.1, "random"),
    ]
    assert all(row.dataset == "gsm8k" for row in rows)
