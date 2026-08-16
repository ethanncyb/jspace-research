# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from jlens.analysis import (
    build_model_summary,
    compare_models,
    spearman,
    steering_case_scores,
    summarize_behavior,
    summarize_steering,
    write_comparison_markdown,
)


def _steering_rows():
    rows = []
    for case in ("a", "b"):
        rows.append(
            {
                "dataset": "gsm8k",
                "case_id": case,
                "layer": 1,
                "strength": 0.0,
                "control": "local_jacobian",
                "target_logprob_lift": 0.0,
                "steered_top1": False,
                "kl_divergence": 0.0,
            }
        )
        for strength in (0.1, 0.2):
            for layer in (1, 2):
                rows.extend(
                    [
                        {
                            "dataset": "gsm8k",
                            "case_id": case,
                            "layer": layer,
                            "strength": strength,
                            "control": "local_jacobian",
                            "target_logprob_lift": strength * 2,
                            "steered_top1": strength == 0.2,
                            "kl_divergence": strength,
                        },
                        {
                            "dataset": "gsm8k",
                            "case_id": case,
                            "layer": layer,
                            "strength": strength,
                            "control": "random",
                            "target_logprob_lift": 0.0,
                            "steered_top1": False,
                            "kl_divergence": strength,
                        },
                    ]
                )
    return rows


def test_behavior_and_steering_tracks_aggregate_separately():
    behavior = summarize_behavior(
        [
            {"dataset": "humaneval", "passed": True, "truncated": False},
            {"dataset": "humaneval", "passed": False, "truncated": True},
            {"dataset": "gsm8k", "passed": True, "truncated": False, "parsed": True},
        ]
    )
    assert behavior["humaneval"]["score"] == 0.5
    assert behavior["gsm8k"]["parse_rate"] == 1
    scores = steering_case_scores(_steering_rows())
    assert set(scores) == {"gsm8k:a", "gsm8k:b"}
    summary = summarize_steering(_steering_rows(), n_resamples=100)
    assert summary["random_adjusted_logprob_auc_ci_low"] > 0
    assert summary["kl_matched_logprob_lift_ci_low"] > 0
    assert summary["steerable"]


def test_spearman_and_cross_model_report_are_explicitly_exploratory():
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1
    models = []
    for size in (4, 8, 14, 32):
        models.append(
            {
                "model": f"qwen3-{size}b",
                "parameters_b": size,
                "benchmark": {
                    "humaneval": {"score": size / 40},
                    "gsm8k": {"score": size / 40},
                },
                "steering": {
                    "random_adjusted_logprob_auc": size / 100,
                    "random_adjusted_logprob_auc_ci_low": size / 110,
                    "kl_matched_logprob_lift": size / 120,
                },
                "security": {
                    "paired_balanced_accuracy": 0.9,
                    "paired_balanced_accuracy_ci_low": 0.7,
                    "attack_success_rate": 1 / size,
                    "mean_clean_utility": 0.8,
                },
                "probe": {"mean_auroc": 0.8 + size / 1000},
            }
        )
    report = compare_models(models)
    assert report["exploratory"]
    assert report["recognition_threshold"] == "qwen3-4b"
    assert report["spearman_vs_log_parameters"]["steerability_auc"] == 1


def test_comparison_markdown_is_human_readable(tmp_path):
    models = []
    for size in (4, 8):
        models.append(
            {
                "model": f"qwen3-{size}b",
                "parameters_b": size,
                "benchmark": {
                    "humaneval": {"score": 0.5},
                    "gsm8k": {"score": 0.6},
                },
                "steering": {
                    "random_adjusted_logprob_auc": 0.1,
                    "random_adjusted_logprob_auc_ci_low": 0.05,
                    "kl_matched_logprob_lift": 0.2,
                },
                "security": {
                    "paired_balanced_accuracy": 0.9,
                    "paired_balanced_accuracy_ci_low": 0.7,
                    "attack_success_rate": 0.1,
                    "mean_clean_utility": 0.8,
                },
                "probe": {"mean_auroc": 0.85},
            }
        )
    report = compare_models(models)
    path = write_comparison_markdown(tmp_path / "comparison.md", report)
    text = path.read_text()
    assert "Exploratory only" in text
    assert "qwen3-4b" in text


def test_model_summary_keeps_all_three_result_families():
    security = {
        "paired_balanced_accuracy": 1.0,
        "paired_balanced_accuracy_ci_low": 0.9,
        "attack_success_rate": 0.5,
    }
    summary = build_model_summary(
        model="qwen3-4b",
        parameters_b=4,
        behavior_rows=[],
        steering_rows=_steering_rows(),
        security_summary=security,
        probe_rows=[{"auroc": 0.9, "layer": 1}],
    )
    assert set(summary) == {"model", "parameters_b", "benchmark", "steering", "security", "probe"}
