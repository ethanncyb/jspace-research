"""Paired comparison of two completed GSM8K runs."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any

from gsm8k_jspace import SCHEMA_VERSION
from gsm8k_jspace.artifacts.writer import read_json, read_jsonl, write_json


CHANGE_TYPES = (
    "correct_both",
    "incorrect_both",
    "broken",
    "fixed",
)


class ComparisonError(ValueError):
    """Incompatible run manifests."""


def _results_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row["example_id"]: row for row in read_jsonl(path)}


def _check_compatible(baseline_manifest: dict, candidate_manifest: dict) -> list[str]:
    warnings: list[str] = []
    if baseline_manifest.get("config_fingerprint") != candidate_manifest.get(
        "config_fingerprint"
    ):
        warnings.append(
            "run fingerprints differ; comparison is valid only if scientific "
            "settings still match"
        )
    if baseline_manifest.get("test_type") != candidate_manifest.get("test_type"):
        raise ComparisonError("test_type mismatch")
    if baseline_manifest.get("dataset_selection_fingerprint") != candidate_manifest.get(
        "dataset_selection_fingerprint"
    ):
        warnings.append("dataset selection fingerprints differ")
    return warnings


def _mcnemar_exact(broken: int, fixed: int) -> float:
    n = broken + fixed
    if n == 0:
        return 1.0
    k = min(broken, fixed)
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i)
    return min(1.0, 2.0 * total / (2 ** n))


def _bootstrap_diff(
    baseline: list[bool],
    candidate: list[bool],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(baseline)
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(baseline[i] for i in idx) / n
        c = sum(candidate[i] for i in idx) / n
        diffs.append(c - b)
    diffs.sort()
    mean = sum(diffs) / len(diffs)
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    return mean, lo, hi


def compare_runs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    base_manifest = read_json(baseline_dir / "manifest.json")
    cand_manifest = read_json(candidate_dir / "manifest.json")
    warnings = _check_compatible(base_manifest, cand_manifest)

    base_results = _results_by_id(baseline_dir / "evaluation" / "results.jsonl")
    cand_results = _results_by_id(candidate_dir / "evaluation" / "results.jsonl")
    shared = sorted(set(base_results) & set(cand_results))
    missing = sorted(set(base_results) ^ set(cand_results))

    counts = {name: 0 for name in CHANGE_TYPES}
    rows: list[dict[str, Any]] = []
    base_correct: list[bool] = []
    cand_correct: list[bool] = []
    for example_id in shared:
        b = bool(base_results[example_id]["correct"])
        c = bool(cand_results[example_id]["correct"])
        if b and c:
            change = "correct_both"
        elif not b and not c:
            change = "incorrect_both"
        elif b and not c:
            change = "broken"
        else:
            change = "fixed"
        counts[change] += 1
        base_correct.append(b)
        cand_correct.append(c)
        rows.append(
            {
                "example_id": example_id,
                "baseline_correct": b,
                "candidate_correct": c,
                "baseline_pred": base_results[example_id].get(
                    "predicted_answer_normalized"
                ),
                "candidate_pred": cand_results[example_id].get(
                    "predicted_answer_normalized"
                ),
                "change_type": change,
            }
        )

    n = len(shared)
    p_base = (sum(base_correct) / n) if n else 0.0
    p_cand = (sum(cand_correct) / n) if n else 0.0
    mean_diff, lo, hi = _bootstrap_diff(base_correct, cand_correct)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "baseline_run": str(baseline_dir),
        "candidate_run": str(candidate_dir),
        "n_shared": n,
        "baseline_accuracy": p_base,
        "candidate_accuracy": p_cand,
        "absolute_change": p_cand - p_base,
        "relative_change": ((p_base - p_cand) / p_base) if p_base else 0.0,
        "counts": counts,
        "mcnemar_exact_p": _mcnemar_exact(counts["broken"], counts["fixed"]),
        "bootstrap_diff_mean": mean_diff,
        "bootstrap_diff_ci95": [lo, hi],
        "missing_or_extra": missing,
        "warnings": warnings,
    }

    if out_dir is None:
        out_dir = candidate_dir / "comparisons" / baseline_dir.name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "paired_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "baseline_correct",
                "candidate_correct",
                "baseline_pred",
                "candidate_pred",
                "change_type",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    print(
        f"[compare] baseline={p_base:.4f} candidate={p_cand:.4f} "
        f"delta={p_cand - p_base:+.4f}"
    )
    return summary


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# GSM8K paired comparison",
        "",
        f"- Baseline: `{summary['baseline_run']}`",
        f"- Candidate: `{summary['candidate_run']}`",
        f"- Shared examples: {summary['n_shared']}",
        f"- Baseline accuracy: {summary['baseline_accuracy']:.4f}",
        f"- Candidate accuracy: {summary['candidate_accuracy']:.4f}",
        f"- Absolute change: {summary['absolute_change']:+.4f}",
        f"- McNemar exact p: {summary['mcnemar_exact_p']:.4g}",
        f"- Bootstrap 95% CI: {summary['bootstrap_diff_ci95']}",
        "",
        "| change_type | n |",
        "|---|---|",
    ]
    for name in CHANGE_TYPES:
        lines.append(f"| {name} | {summary['counts'][name]} |")
    if summary["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- {item}" for item in summary["warnings"]]
    if summary["missing_or_extra"]:
        lines += ["", "## Missing or extra example IDs", ""]
        lines += [f"- `{item}`" for item in summary["missing_or_extra"]]
    lines += [
        "",
        "Paired causal interpretation requires the same hardware/backend, prompt,",
        "dataset selection, and generation settings.",
        "",
    ]
    return "\n".join(lines)
