"""Shared evaluation writing helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from gsm8k_jspace.artifacts.writer import write_json, write_jsonl


def write_evaluation(
    run_dir: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    report_md: str,
) -> dict[str, Any]:
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(eval_dir / "results.jsonl", results)
    write_json(eval_dir / "summary.json", summary)
    (eval_dir / "report.md").write_text(report_md, encoding="utf-8")
    return summary


def generation_stats(completions: list[dict[str, Any]]) -> dict[str, Any]:
    finish_reasons: Counter[str] = Counter()
    token_counts: list[int] = []
    for rec in completions:
        finish_reasons[str(rec.get("finish_reason", "unknown"))] += 1
        token_counts.append(int(rec.get("n_generated_tokens", 0)))
    return {
        "finish_reasons": dict(finish_reasons),
        "mean_generated_tokens": (
            sum(token_counts) / len(token_counts) if token_counts else 0.0
        ),
        "median_generated_tokens": (
            sorted(token_counts)[len(token_counts) // 2] if token_counts else 0
        ),
    }
