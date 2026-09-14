"""Portable plotting data and offline, filterable Phase 2 comparisons.

This module deliberately depends only on CPU data/plotting libraries. Redrawing
an export never loads a model, a Phase 1 handoff, or a judge client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from ..runtime import atomic_write_json, read_json, sha256_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCHEMA_VERSION = 1
DIMENSION_NAMES = {"K": "sparsity_k", "W": "output_token_window"}
OVERVIEW_METRICS = (
    ("asr", "attack", "Attack success rate", 1.0),
    ("garbage_rate", "control", "Clean-output garbage rate", 1.0),
    ("mean_degradation_severity", "control", "Clean-output severity", 3.0),
)
METRIC_DEFINITIONS = {
    "asr": "Fraction of attack outputs judged YES. Attack-judge UNKNOWN counts as unsuccessful.",
    "garbage_rate": "Fraction of determinate quality judgments labeled YES (severity 3). UNKNOWN excluded.",
    "quality_unknown_rate": "Fraction of all outputs whose quality judgment is UNKNOWN.",
    "mean_degradation_severity": "Mean severity (0 coherent through 3 unusable), excluding UNKNOWN.",
    "refusal_rate": "Fraction of outputs matching the fixed refusal-prefix detector.",
    "rouge1_recall": "ROUGE-1 recall against the frozen task reference, reported per task.",
    "rouge2_recall": "ROUGE-2 recall against the frozen task reference, reported per task.",
    "rougeL_recall": "ROUGE-L recall against the frozen task reference, reported per task.",
    "rougeLsum_recall": "ROUGE-Lsum recall against the frozen task reference, reported per task.",
    **{
        f"severity_{i}_rate": f"Fraction of determinate quality judgments with severity {i}."
        for i in range(4)
    },
}


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rows(frame: pd.DataFrame):
    columns = [DIMENSION_NAMES.get(str(column), str(column)) for column in frame.columns]
    for values in frame.itertuples(index=False, name=None):
        yield {key: _json_value(value) for key, value in zip(columns, values, strict=True)}


def _write_detail_jsonl(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in _rows(frame):
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_plot_data(
    output_dir: Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    provenances: list[dict[str, Any]],
    *,
    label: str,
) -> Path:
    """Export exact numeric summaries and one JSON object per scored example."""
    detail_path = output_dir / "phase2_results.jsonl"
    _write_detail_jsonl(detail_path, results)
    first = provenances[0]
    child_ids = sorted(p["run_id"] for p in provenances)
    run_id = "phase2-sweep-" + hashlib.sha256(json.dumps(child_ids).encode()).hexdigest()[:16]
    metadata = {
        key: first.get(key)
        for key in ("model", "lens", "seed", "manifest_sha256", "judge", "quality_judge")
    }
    metadata.update(
        run_id=run_id,
        label=label,
        intervention=first["resolved_config"]["intervention"],
        selected_layers={
            str(int(row.K)): int(row.selected_layer)
            for row in results[["K", "selected_layer"]].drop_duplicates().itertuples()
        },
        generation={
            key: first["resolved_config"][key]
            for key in ("max_new_tokens", "do_sample", "generation_batch_size")
        },
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "phase2_plot_data",
        "run": metadata,
        "dimensions": {
            "sparsity_k_values": sorted(int(k) for k in results.K.unique()),
            "output_token_windows": sorted(int(w) for w in results.W.unique()),
            "alphas": sorted(float(a) for a in results.alpha.unique()),
            "tasks": sorted(str(t) for t in results.task.unique()),
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "field_definitions": {
            "n": "Metric denominator; quality metrics exclude UNKNOWN except quality_unknown_rate.",
            "n_total": "Total outputs in a quality group; null for other metrics (use n).",
            "n_unknown": "Number of indeterminate quality judgments; null for other metrics.",
            "baseline_value": "Same metric at alpha=0 within this run, sparsity, window, task, and condition.",
            "delta": "value minus baseline_value; null when either is undefined.",
            "retention": "Utility divided by its own alpha=0 baseline; null when baseline is zero or metric is not utility.",
            "value": "Observed metric value. JSON null denotes unavailable/undefined, never zero.",
        },
        "summary": list(_rows(summary)),
        "per_example": {
            "path": detail_path.name,
            "format": "jsonl",
            "records": len(results),
            "sha256": sha256_file(detail_path),
        },
        "provenance": provenances,
    }
    path = output_dir / "phase2_plot_data.json"
    atomic_write_json(path, payload)
    return path


def load_plot_data(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "phase2_plot_data.json"
    payload = read_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != "phase2_plot_data":
        raise ValueError(f"Unsupported Phase 2 plotting export: {path}")
    required = {
        "sparsity_k",
        "output_token_window",
        "alpha",
        "metric",
        "scope",
        "condition",
        "task",
        "value",
        "n",
    }
    if (
        not isinstance(payload.get("run"), dict)
        or not isinstance(payload.get("summary"), list)
        or not payload["summary"]
    ):
        raise ValueError(f"Incomplete plotting export: {path}")
    for row in payload["summary"]:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Incomplete summary row at {path}")
        if any(
            type(row[key]) is not int or row[key] <= 0
            for key in ("sparsity_k", "output_token_window")
        ):
            raise ValueError(f"Invalid combination at {path}")
        if type(row["alpha"]) not in (int, float) or not math.isfinite(row["alpha"]):
            raise ValueError(f"Invalid alpha at {path}")
        if row["value"] is not None and (
            type(row["value"]) not in (int, float) or not math.isfinite(row["value"])
        ):
            raise ValueError(f"Invalid metric value at {path}")
    frame = pd.DataFrame(payload["summary"])
    if frame.duplicated(
        ["sparsity_k", "output_token_window", "alpha", "metric", "scope", "condition", "task"]
    ).any():
        raise ValueError(f"Duplicate metric cells at {path}")
    return payload


def select_summary(
    payload: dict[str, Any],
    *,
    sparsity_k_values: Sequence[int] | None = None,
    output_token_windows: Sequence[int] | None = None,
    alphas: Sequence[float] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(payload["summary"])
    for column, values in (
        ("sparsity_k", sparsity_k_values),
        ("output_token_window", output_token_windows),
        ("alpha", alphas),
    ):
        if values is not None:
            frame = frame[frame[column].isin(values)]
    return frame


def metric_matrix(
    frame: pd.DataFrame,
    metric: str,
    condition: str,
    combinations: list[tuple[int, int]],
    alphas: list[float],
    *,
    task: str | None = None,
) -> np.ndarray:
    rows = frame[(frame.metric == metric) & (frame.condition == condition)]
    rows = (
        rows[rows.scope == "overall"]
        if task is None
        else rows[(rows.scope == "task") & (rows.task == task)]
    )
    values = {
        (int(r.sparsity_k), int(r.output_token_window), float(r.alpha)): r.value
        for r in rows.itertuples()
    }
    return np.array(
        [
            [
                np.nan if values.get((k, w, a)) is None else values.get((k, w, a), np.nan)
                for a in alphas
            ]
            for k, w in combinations
        ],
        dtype=float,
    )


def _draw_matrix(
    axis: Any,
    values: np.ndarray,
    combinations: list[tuple[int, int]],
    alphas: list[float],
    title: str,
    maximum: float,
    *,
    utility: bool = False,
) -> Any:
    palette = plt.get_cmap("YlGn" if utility else "YlOrRd").copy()
    palette.set_bad("#e5e7eb")
    chart = axis.imshow(
        np.ma.masked_invalid(values), aspect="auto", vmin=0, vmax=maximum, cmap=palette
    )
    axis.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas])
    axis.set_yticks(range(len(combinations)), [f"K={k}, W={w}" for k, w in combinations])
    axis.set_xlabel("Alpha")
    axis.set_title(title, fontsize=10)
    for row in range(len(combinations)):
        for col in range(len(alphas)):
            value = values[row, col]
            axis.text(
                col,
                row,
                "N/A" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if np.isfinite(value) and value / maximum > 0.65 else "black",
            )
    return chart


def _heatmaps(
    frames: list[pd.DataFrame],
    labels: list[str],
    output_dir: Path,
    *,
    comparison: bool,
    note: str,
) -> list[Path]:
    all_rows = pd.concat(frames, ignore_index=True)
    combinations = sorted(
        {(int(r.sparsity_k), int(r.output_token_window)) for r in all_rows.itertuples()}
    )
    alphas = sorted(float(a) for a in all_rows.alpha.unique())
    tasks = sorted(
        str(t) for t in all_rows[all_rows.metric == "rougeL_recall"].task.dropna().unique()
    )
    outputs = []
    panels = [(None, list(OVERVIEW_METRICS))]
    if tasks:
        panels.append(("utility", [(task, "control", task, 1.0) for task in tasks]))
    for kind, metrics in panels:
        figure, axes = plt.subplots(
            len(frames),
            len(metrics),
            squeeze=False,
            figsize=(
                max((0.65 * len(alphas) + 2) * len(metrics), 5 * len(metrics), 10),
                max(3.4, 0.3 * len(combinations) + 1.6) * len(frames),
            ),
        )
        for row, (frame, label) in enumerate(zip(frames, labels, strict=True)):
            for column, (metric, condition, title, maximum) in enumerate(metrics):
                values = metric_matrix(
                    frame,
                    "rougeL_recall" if kind == "utility" else metric,
                    condition,
                    combinations,
                    alphas,
                    task=metric if kind == "utility" else None,
                )
                chart = _draw_matrix(
                    axes[row, column],
                    values,
                    combinations,
                    alphas,
                    f"{label}\n{title}",
                    maximum,
                    utility=kind == "utility",
                )
                figure.colorbar(chart, ax=axes[row, column], fraction=0.04, pad=0.03)
        description = (
            "Clean utility by task — higher is better"
            if kind == "utility"
            else "Intervention overview — lower is better in each panel"
        )
        figure.suptitle(
            description + "\nN/A = unmeasured or undefined; no interpolation", fontsize=12
        )
        figure.text(0.01, 0.005, note, fontsize=8)
        figure.tight_layout(rect=(0, 0.03, 1, 0.94))
        stem = "phase2_run_comparison" if comparison else "phase2_run_overview"
        path = output_dir / (stem + ("_utility" if kind == "utility" else "") + ".png")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)
    return outputs


def save_run_overview(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    return _heatmaps(
        [pd.DataFrame(payload["summary"])],
        [payload["run"]["label"]],
        output_dir,
        comparison=False,
        note="Read garbage rate alongside its UNKNOWN rate and denominator in phase2_plot_data.json.",
    )


def _differences(payloads: list[dict[str, Any]]) -> list[str]:
    fields = (
        "model",
        "lens",
        "seed",
        "manifest_sha256",
        "intervention",
        "generation",
        "judge",
        "quality_judge",
    )
    different = [
        field
        for field in fields
        if len({json.dumps(p["run"].get(field), sort_keys=True) for p in payloads}) > 1
    ]
    if len({json.dumps(p["run"].get("selected_layers"), sort_keys=True) for p in payloads}) > 1:
        different.append("selected_layers")
    if (
        len({json.dumps(p.get("dimensions", {}).get("tasks"), sort_keys=True) for p in payloads})
        > 1
    ):
        different.append("tasks")
    return different


def plot_exports(
    paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    labels: Sequence[str] | None = None,
    sparsity_k_values: Sequence[int] | None = None,
    output_token_windows: Sequence[int] | None = None,
    alphas: Sequence[float] | None = None,
    metric: str = "asr",
    condition: str = "attack",
    task: str | None = None,
) -> list[Path]:
    if not paths:
        raise ValueError("At least one plotting export is required")
    payloads = [load_plot_data(path) for path in paths]
    names = (
        list(labels)
        if labels is not None
        else [
            f"{i + 1}: {p['run'].get('label', p['run'].get('run_id'))}"
            for i, p in enumerate(payloads)
        ]
    )
    if len(names) != len(payloads) or len(set(names)) != len(names):
        raise ValueError("Provide one unique label per run")
    frames = [
        select_summary(
            p,
            sparsity_k_values=sparsity_k_values,
            output_token_windows=output_token_windows,
            alphas=alphas,
        )
        for p in payloads
    ]
    if all(frame.empty for frame in frames):
        raise ValueError("No combinations match the filters in any run")
    selected = []
    for frame in frames:
        rows = frame[(frame.metric == metric) & (frame.condition == condition)]
        selected.append(
            rows[rows.scope == "overall"]
            if task is None
            else rows[(rows.scope == "task") & (rows.task == task)]
        )
    if all(rows.empty for rows in selected):
        raise ValueError("No selected curve data; task-specific metrics require --task")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    differences = _differences(payloads)
    note = (
        "Descriptive comparison. Different settings: " + ", ".join(differences)
        if differences
        else "Same dataset identity and scoring settings. Runs are plotted separately."
    )
    if len(payloads) == 1:
        note = "Single-run selection. Missing combinations remain N/A."
    outputs = _heatmaps(frames, names, directory, comparison=len(payloads) > 1, note=note)
    combinations = sorted(
        {
            (int(r.sparsity_k), int(r.output_token_window))
            for frame in selected
            for r in frame.itertuples()
        }
    )
    figure, axes = plt.subplots(
        math.ceil(len(combinations) / 3),
        min(3, len(combinations)),
        squeeze=False,
        figsize=(5 * min(3, len(combinations)), 3.8 * math.ceil(len(combinations) / 3)),
    )
    for axis, (k, w) in zip(axes.flat, combinations, strict=False):
        for label, frame in zip(names, selected, strict=True):
            rows = frame[(frame.sparsity_k == k) & (frame.output_token_window == w)].sort_values(
                "alpha"
            )
            axis.plot(
                rows.alpha, pd.to_numeric(rows.value, errors="coerce"), marker="o", label=label
            )
        axis.set_title(f"K={k}, W={w}" + (f", {task}" if task else ""))
        axis.set_xlabel("Alpha")
        axis.set_ylabel(metric)
        axis.legend(fontsize="small")
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(combinations) :]:
        axis.set_visible(False)
    figure.suptitle(f"{condition}: {metric}")
    figure.text(0.01, 0.005, note, fontsize=8)
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    curves = directory / "phase2_selected_curves.png"
    figure.savefig(curves, dpi=180, bbox_inches="tight")
    plt.close(figure)
    outputs.append(curves)
    atomic_write_json(
        directory / "phase2_plot_selection.json",
        {
            "schema_version": SCHEMA_VERSION,
            "filters": {
                "sparsity_k_values": sparsity_k_values,
                "output_token_windows": output_token_windows,
                "alphas": alphas,
                "metric": metric,
                "condition": condition,
                "task": task,
            },
            "comparison_differences": differences,
            "runs": [
                {
                    "label": label,
                    "source": str(Path(path).resolve()),
                    "run": payload["run"],
                    "summary": list(_rows(frame)),
                }
                for path, label, payload, frame in zip(paths, names, payloads, frames, strict=True)
            ],
        },
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redraw Phase 2 JSON exports or compare runs without model/API access."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Phase 2 directories or phase2_plot_data.json files",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", nargs="+", help="One unique display label per run")
    parser.add_argument("--sparsity-k-values", nargs="+", type=int)
    parser.add_argument("--output-token-windows", nargs="+", type=int)
    parser.add_argument("--alphas", nargs="+", type=float)
    parser.add_argument(
        "--metric", default="asr", help="Summary metric, e.g. asr, garbage_rate, rougeL_recall"
    )
    parser.add_argument("--condition", choices=("attack", "control"), default="attack")
    parser.add_argument(
        "--task", help="Required for task-specific utility curves; otherwise plot overall values"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for path in plot_exports(
        args.runs,
        args.output_dir,
        labels=args.labels,
        sparsity_k_values=args.sparsity_k_values,
        output_token_windows=args.output_token_windows,
        alphas=args.alphas,
        metric=args.metric,
        condition=args.condition,
        task=args.task,
    ):
        print(path)


if __name__ == "__main__":
    main()
