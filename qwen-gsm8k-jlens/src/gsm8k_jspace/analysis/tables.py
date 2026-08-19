"""Tabular summaries for notebooks and reports."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from gsm8k_jspace.analysis.loaders import RunArtifacts, iter_capture_rows


def accuracy_table(run: RunArtifacts) -> list[dict[str, Any]]:
    if run.summary:
        return [
            {
                "run_id": run.summary.get("run_id"),
                "condition": run.summary.get("condition"),
                "accuracy": run.summary.get("accuracy"),
                "n_correct": run.summary.get("n_correct"),
                "n_evaluated": run.summary.get("n_evaluated"),
                "extraction_rate": run.summary.get("extraction_rate"),
            }
        ]
    return []


def capture_layer_table(
    run: RunArtifacts, *, max_rows: int = 250_000
) -> list[dict[str, Any]]:
    acc: dict[int, dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "hidden_norm": 0.0, "jspace_norm": 0.0}
    )
    for row in iter_capture_rows(run.run_dir, max_rows=max_rows):
        layer = int(row["layer"])
        acc[layer]["n"] += 1
        acc[layer]["hidden_norm"] += float(row.get("hidden_norm") or 0.0)
        acc[layer]["jspace_norm"] += float(row.get("jspace_norm") or 0.0)
    table = []
    for layer in sorted(acc):
        n = max(acc[layer]["n"], 1.0)
        table.append(
            {
                "layer": layer,
                "n": int(acc[layer]["n"]),
                "mean_hidden_norm": acc[layer]["hidden_norm"] / n,
                "mean_jspace_norm": acc[layer]["jspace_norm"] / n,
            }
        )
    return table
