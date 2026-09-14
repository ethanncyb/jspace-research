from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_phase2_scoring import make_results

from jspace_research.phase2.plots import (
    build_parser,
    export_plot_data,
    load_plot_data,
    metric_matrix,
    plot_exports,
    save_run_overview,
    select_summary,
)
from jspace_research.phase2.scoring import summarize_quality, summarize_results


def make_export(root: Path, *, label="test run", manifest="manifest-a") -> Path:
    first = make_results().assign(
        K=20, W=1, selected_layer=2, garbage_label="NO", degradation_severity=0
    )
    second = make_results().assign(
        K=30, W=5, selected_layer=4, garbage_label="YES", degradation_severity=3
    )
    # One completely indeterminate quality group tests null/NaN handling.
    second.loc[second.alpha == 1.0, "garbage_label"] = "UNKNOWN"
    second.loc[second.alpha == 1.0, "degradation_severity"] = np.nan
    results = pd.concat([first, second], ignore_index=True)
    results["quality_explanation"] = "Synthetic test output."
    results["is_valid"] = pd.array([None] * len(results), dtype="boolean")
    summary = pd.concat([summarize_results(results), summarize_quality(results)], ignore_index=True)
    provenance = {
        "run_id": "child-run",
        "model": {"id": "test-model"},
        "lens": {"sha256": "lens"},
        "manifest_sha256": manifest,
        "seed": 42,
        "judge": {"rubric_sha256": "attack-rubric"},
        "quality_judge": {"rubric_sha256": "quality-rubric"},
        "resolved_config": {
            "intervention": "first_output_tokens_per_token_reconstruction_v1",
            "max_new_tokens": 512,
            "do_sample": False,
            "generation_batch_size": 1,
        },
    }
    return export_plot_data(root, results, summary, [provenance], label=label)


def test_export_has_all_values_denominators_and_per_example_json(tmp_path):
    path = make_export(tmp_path)
    raw = path.read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    payload = load_plot_data(path)
    assert payload["dimensions"]["sparsity_k_values"] == [20, 30]
    assert payload["dimensions"]["output_token_windows"] == [1, 5]
    assert payload["run"]["selected_layers"] == {"20": 2, "30": 4}
    assert payload["metric_definitions"]["garbage_rate"]
    assert payload["field_definitions"]["baseline_value"]
    detail = [
        json.loads(line)
        for line in (tmp_path / payload["per_example"]["path"]).read_text().splitlines()
    ]
    assert len(detail) == payload["per_example"]["records"] == 16
    assert detail[0]["sparsity_k"] == 20
    assert detail[0]["output_token_window"] == 1
    assert detail[0]["is_valid"] is None
    assert detail[0]["quality_explanation"] == "Synthetic test output."
    rows = select_summary(payload, sparsity_k_values=[30], alphas=[1.0])
    unknown = rows[
        (rows.metric == "garbage_rate") & (rows.scope == "overall") & (rows.condition == "control")
    ].iloc[0]
    assert pd.isna(unknown.value) and unknown.n == 0 and unknown.n_unknown == 2
    assert unknown.n_total == 2


def test_heatmaps_align_by_values_and_do_not_fill_missing_cells(tmp_path):
    payload = load_plot_data(make_export(tmp_path))
    matrix = metric_matrix(
        select_summary(payload),
        "garbage_rate",
        "control",
        [(20, 1), (20, 5), (30, 5)],
        [0.0, 0.5, 1.0],
    )
    np.testing.assert_allclose(
        matrix, [[0, np.nan, 0], [np.nan, np.nan, np.nan], [1, np.nan, np.nan]], equal_nan=True
    )
    outputs = save_run_overview(payload, tmp_path)
    assert {p.name for p in outputs} == {
        "phase2_run_overview.png",
        "phase2_run_overview_utility.png",
    }
    assert all(path.stat().st_size > 1000 for path in outputs)


def test_offline_filtering_and_comparison_preserve_run_identity(tmp_path):
    first = make_export(tmp_path / "first", label="first")
    second = make_export(tmp_path / "second", label="second", manifest="different population")
    output = tmp_path / "plots"
    files = plot_exports(
        [first, second], output, labels=["A", "B"], sparsity_k_values=[20], output_token_windows=[1]
    )
    assert {path.name for path in files} == {
        "phase2_run_comparison.png",
        "phase2_run_comparison_utility.png",
        "phase2_selected_curves.png",
    }
    selection = json.loads((output / "phase2_plot_selection.json").read_text())
    assert selection["comparison_differences"] == ["manifest_sha256"]
    assert [run["label"] for run in selection["runs"]] == ["A", "B"]
    for run in selection["runs"]:
        assert {row["sparsity_k"] for row in run["summary"]} == {20}
        assert {row["output_token_window"] for row in run["summary"]} == {1}
        assert run["source"]
    # Utility curves require task selection and do not average across tasks.
    plot_exports(
        [first], tmp_path / "utility", metric="rougeL_recall", condition="control", task="email"
    )
    with pytest.raises(ValueError, match="task-specific"):
        plot_exports([first], tmp_path / "bad", metric="rougeL_recall", condition="control")
    with pytest.raises(ValueError, match="No combinations"):
        plot_exports([first], tmp_path / "empty", sparsity_k_values=[99])


def test_plot_loader_rejects_duplicate_or_nonfinite_cells(tmp_path):
    path = make_export(tmp_path)
    payload = json.loads(path.read_text())
    payload["summary"].append(payload["summary"][0])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Duplicate"):
        load_plot_data(path)
    payload["summary"].pop()
    payload["summary"][0]["value"] = float("inf")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid metric"):
        load_plot_data(path)


def test_plot_cli_filters_use_descriptive_names():
    args = build_parser().parse_args(
        [
            "--runs",
            "run-a/phase2",
            "run-b/phase2",
            "--output-dir",
            "plots",
            "--sparsity-k-values",
            "20",
            "30",
            "--output-token-windows",
            "1",
            "5",
            "--alphas",
            "0",
            "0.5",
            "--labels",
            "A",
            "B",
        ]
    )
    assert args.sparsity_k_values == [20, 30]
    assert args.output_token_windows == [1, 5]
    assert args.alphas == [0.0, 0.5]
