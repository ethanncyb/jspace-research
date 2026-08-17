"""Offline smoke tests for CSV-only visualization generation."""

from __future__ import annotations

import csv

from visualize import generate_all_visuals


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_generate_all_visuals_from_existing_csv_schemas(tmp_path):
    eval_csv = tmp_path / "eval.csv"
    _write_csv(
        eval_csv,
        [
            {
                "label": 0,
                "category": "benign",
                "intervention_triggered": 0,
                "intervention_mode": "none",
                "layer_3_score": 0.1,
                "layer_7_score": 0.2,
            },
            {
                "label": 1,
                "category": "injected_or_jailbreak",
                "intervention_triggered": 1,
                "intervention_mode": "circuit_breaker",
                "layer_3_score": 0.3,
                "layer_7_score": 0.8,
            },
            {
                "label": 1,
                "category": "injected_or_jailbreak",
                "intervention_triggered": 1,
                "intervention_mode": "hard_stop",
                "layer_3_score": 0.6,
                "layer_7_score": 0.9,
            },
        ],
    )
    rounds_csv = tmp_path / "rounds.csv"
    _write_csv(
        rounds_csv,
        [
            {
                "round": 1,
                "attacker_asr": 0.6,
                "probe_unsafe_recall": 0.7,
                "probe_benign_pass_rate": 0.95,
            },
            {
                "round": 2,
                "attacker_asr": 0.4,
                "probe_unsafe_recall": 0.85,
                "probe_benign_pass_rate": 0.92,
            },
        ],
    )
    attacks_csv = tmp_path / "attacks.csv"
    _write_csv(
        attacks_csv,
        [
            {"round": 1, "strategy": "roleplay"},
            {"round": 1, "strategy": "encoding"},
            {"round": 2, "strategy": "encoding"},
        ],
    )
    evolution_cross_csv = tmp_path / "evolution_cross.csv"
    _write_csv(
        evolution_cross_csv,
        [
            {
                "attack_round": 1,
                "guard_round": 1,
                "attack_family": "evolved",
                "asr": 0.5,
            },
            {
                "attack_round": 1,
                "guard_round": 2,
                "attack_family": "evolved",
                "asr": 0.25,
            },
            {
                "attack_round": "heldout",
                "guard_round": 1,
                "attack_family": "heldout_untrained",
                "asr": 0.54,
            },
            {
                "attack_round": "heldout",
                "guard_round": 2,
                "attack_family": "heldout_untrained",
                "asr": 0.26,
            },
        ],
    )
    benchmark_asr_csv = tmp_path / "asr.csv"
    _write_csv(
        benchmark_asr_csv,
        [
            {
                "guard_round": 1,
                "method": "evolved_pool",
                "source_dataset": "all",
                "asr": 0.5,
            },
            {
                "guard_round": 1,
                "method": "renellm",
                "source_dataset": "all",
                "asr": 0.3,
            },
        ],
    )
    benchmark_cross_csv = tmp_path / "benchmark_cross.csv"
    _write_csv(
        benchmark_cross_csv,
        [
            {
                "attack_round": 0,
                "guard_round": 0,
                "source_dataset": "all",
                "asr": 0.8,
            },
            {
                "attack_round": 1,
                "guard_round": 0,
                "source_dataset": "all",
                "asr": 0.6,
            },
        ],
    )

    charts = generate_all_visuals(
        eval_csv=eval_csv,
        round_metrics_csv=rounds_csv,
        attacks_csv=attacks_csv,
        evolution_cross_csv=evolution_cross_csv,
        benchmark_asr_csv=benchmark_asr_csv,
        benchmark_cross_csv=benchmark_cross_csv,
        output_dir=tmp_path / "charts",
    )

    assert len(charts) == 8
    assert all(chart.exists() and chart.stat().st_size > 0 for chart in charts)
