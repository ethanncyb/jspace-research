"""Evaluate saved completions without loading a model."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from gsm8k_jspace import SCHEMA_VERSION
from gsm8k_jspace.artifacts.writer import read_jsonl, write_json, write_jsonl
from gsm8k_jspace.config import AppConfig, load_config
from gsm8k_jspace.evaluation.answer_parser import answers_match, parse_gsm8k_answer


def _gold_for(
    example_id: str,
    selection: list[dict[str, Any]],
    completion: dict[str, Any],
) -> str:
    by_id = {row["example_id"]: row for row in selection}
    if example_id in by_id:
        return str(by_id[example_id]["gold_answer"])
    if "gold_answer" in completion:
        return str(completion["gold_answer"])
    raise KeyError(f"no gold answer for {example_id}")


def evaluate_run(
    run_dir: str | Path,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    completions = read_jsonl(run_dir / "completions.jsonl")
    selection_path = run_dir / "dataset_selection.jsonl"
    selection = read_jsonl(selection_path) if selection_path.exists() else []
    if cfg is None:
        cfg = load_config(run_dir / "resolved_config.yaml")
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    n_correct = 0
    n_extract_ok = 0
    finish_reasons: Counter[str] = Counter()
    token_counts: list[int] = []
    for rec in completions:
        gold_raw = _gold_for(rec["example_id"], selection, rec)
        predicted = parse_gsm8k_answer(
            rec.get("generated_text", ""),
            marker=cfg.prompt.answer_marker,
            prefer_answer_marker=cfg.evaluation.prefer_answer_marker,
            allow_last_number_fallback=cfg.evaluation.allow_last_number_fallback,
        )
        gold = parse_gsm8k_answer(
            gold_raw,
            marker=cfg.prompt.answer_marker,
            prefer_answer_marker=True,
            allow_last_number_fallback=True,
        )
        correct = answers_match(
            predicted, gold, tolerance=cfg.evaluation.numeric_tolerance
        )
        n_correct += int(correct)
        n_extract_ok += int(predicted.succeeded)
        finish_reasons[str(rec.get("finish_reason", "unknown"))] += 1
        token_counts.append(int(rec.get("n_generated_tokens", 0)))
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": rec.get("run_id"),
                "example_id": rec["example_id"],
                "gold_answer_raw": gold_raw,
                "gold_answer_normalized": gold.normalized,
                "predicted_answer_raw": predicted.raw,
                "predicted_answer_normalized": predicted.normalized,
                "extraction_method": predicted.method,
                "extraction_succeeded": predicted.succeeded,
                "correct": correct,
            }
        )

    n_eval = len(results)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": completions[0]["run_id"] if completions else None,
        "model": completions[0]["model"] if completions else None,
        "condition": completions[0]["condition"] if completions else None,
        "n_selected": len(selection) or n_eval,
        "n_completed": n_eval,
        "n_evaluated": n_eval,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_eval) if n_eval else 0.0,
        "n_extraction_ok": n_extract_ok,
        "extraction_rate": (n_extract_ok / n_eval) if n_eval else 0.0,
        "finish_reasons": dict(finish_reasons),
        "mean_generated_tokens": (
            sum(token_counts) / len(token_counts) if token_counts else 0.0
        ),
        "median_generated_tokens": (
            sorted(token_counts)[len(token_counts) // 2] if token_counts else 0
        ),
        "metric_scope": (
            "gsm8k_exact_answer"
            if cfg.benchmark.test_type == "full_answer"
            else "controllability"
        ),
    }
    write_jsonl(eval_dir / "results.jsonl", results)
    write_json(eval_dir / "summary.json", summary)
    (eval_dir / "report.md").write_text(_report_markdown(summary, results), encoding="utf-8")
    print(f"[evaluate] accuracy={summary['accuracy']:.4f} n={n_eval}")
    return summary


def _report_markdown(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    failed = [row for row in results if not row["correct"]]
    lines = [
        "# GSM8K evaluation",
        "",
        f"- Run: `{summary.get('run_id')}`",
        f"- Model: `{summary.get('model')}`",
        f"- Condition: `{summary.get('condition')}`",
        f"- Metric scope: `{summary.get('metric_scope')}`",
        f"- Accuracy: **{summary['accuracy']:.4f}** ({summary['n_correct']}/{summary['n_evaluated']})",
        f"- Extraction rate: {summary['extraction_rate']:.4f}",
        f"- Mean generated tokens: {summary['mean_generated_tokens']:.1f}",
        "",
        "## Incorrect examples",
        "",
    ]
    if not failed:
        lines.append("_None._")
    else:
        for row in failed:
            lines.append(
                f"- `{row['example_id']}` gold=`{row['gold_answer_normalized']}` "
                f"pred=`{row['predicted_answer_normalized']}` "
                f"extract={row['extraction_method']}"
            )
    return "\n".join(lines) + "\n"
