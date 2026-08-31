from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from jspace_research.phase1.artifacts import Phase1Handoff
from jspace_research.phase1.config import (
    DataConfig,
    DependencyConfig,
    LensConfig,
    ModelConfig,
    Phase1Config,
)
from jspace_research.phase1.jspace import tensor_to_bfloat16_bits
from jspace_research.phase3.artifacts import load_detector
from jspace_research.phase3.config import Phase3Config
from jspace_research.phase3.pipeline import (
    build_sparse_features,
    run,
    select_threshold,
)


def make_config(tmp_path: Path) -> Phase3Config:
    phase1 = Phase1Config(
        model=ModelConfig("model", "a" * 40),
        lens=LensConfig("lens", "b" * 40, "lens.pt", "c" * 64),
        dependencies=DependencyConfig(
            "581d398613e5602a5af361e1c34d3a92ea82ba8e",
            "a004b69ec0dd446e0afd461d98cb5e96e120a5d0",
        ),
        data=DataConfig(tmp_path / "BIPIA" / "benchmark"),
        output_dir=tmp_path / "phase1",
        seed=42,
        tasks=("email",),
        train_pairs_per_task=12,
        validation_pairs_per_task=6,
        max_input_tokens=4096,
        token_match_tolerance=1,
        sparsity_k=25,
        screen_candidates=512,
        decomposition_batch_size=8,
        dictionary_chunk_size=4096,
        smoke_layer_count=6,
    )
    return Phase3Config(
        phase1=phase1,
        phase1_selected_path=tmp_path / "phase1" / "selected_layer.json",
        output_dir=tmp_path / "phase3",
        penalty="l2",
        regularization_c=1.0,
        solver="liblinear",
        fit_intercept=True,
        class_weight=None,
        random_state=42,
        max_iter=1000,
        tol=1e-4,
    )


def fake_handoff(config: Phase3Config) -> Phase1Handoff:
    examples: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        for pair_number in range(2):
            for condition, label in (("attack", 1), ("control", 0)):
                examples.append(
                    {
                        "example_index": len(examples),
                        "pair_id": f"email:{split}:{pair_number:05d}",
                        "task": "email",
                        "task_display": "EmailQA",
                        "split": split,
                        "condition": condition,
                        "label": label,
                    }
                )

    number_examples = len(examples)
    reconstruction = torch.tensor(
        [[1.0, 0.0] if row["label"] else [-1.0, 0.0] for row in examples]
    )
    support_ids = np.full((number_examples, 25), -1, dtype=np.int32)
    coefficients = np.zeros((number_examples, 25), dtype=np.float32)
    for row in examples:
        index = int(row["example_index"])
        support_ids[index, 0] = 10 if row["label"] else 20
        coefficients[index, 0] = 1.0
        if row["split"] == "validation":
            support_ids[index, 1] = 99
            coefficients[index, 1] = 3.0

    return Phase1Handoff(
        metadata={
            "run_id": "phase1-run",
            "config_sha256": config.phase1.identity_hash(),
            "manifest_sha256": "d" * 64,
            "selected_layer": 2,
            "decomposition": {
                "method": "screened_nonnegative_greedy_approximation",
                "dictionary_l2_normalized": True,
                "sparsity_k": 25,
                "screen_candidates": 512,
            },
        },
        direction={
            "layer": 2,
            "mu_clean": torch.tensor([-1.0, 0.0]),
            "d_raw": torch.tensor([2.0, 0.0]),
            "d_unit": torch.tensor([1.0, 0.0]),
            "d_norm": 2.0,
        },
        examples=examples,
        reconstruction=tensor_to_bfloat16_bits(reconstruction),  # type: ignore[arg-type]
        support_ids=support_ids,  # type: ignore[arg-type]
        coefficients=coefficients,  # type: ignore[arg-type]
        reconstruction_shape=(number_examples, 2),
        sparse_shape=(number_examples, 25),
    )


def test_sparse_features_are_defined_only_by_training_supports(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    handoff = fake_handoff(config)
    train = np.array([0, 1, 2, 3])
    validation = np.array([4, 5, 6, 7])
    matrix, feature_ids = build_sparse_features(handoff, train, validation)

    np.testing.assert_array_equal(feature_ids, [10, 20])
    assert matrix.shape == (4, 2)
    assert matrix.nnz == 4
    assert 99 not in feature_ids


def test_threshold_ties_choose_the_higher_candidate() -> None:
    threshold, macro_ba = select_threshold(
        np.array([0.0, 0.0]),
        np.array([0, 1]),
        np.array(["email", "email"], dtype=object),
    )
    assert threshold > 0.0
    assert macro_ba == pytest.approx(0.5)


def test_phase3_cpu_flow_writes_and_loads_only_planned_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    handoff = fake_handoff(config)
    from jspace_research.phase3 import pipeline

    monkeypatch.setattr(pipeline, "load_phase1_handoff", lambda path, phase1: handoff)
    metrics_path = run(config)

    expected = {
        "mean_detector.pt",
        "logistic_detector.pt",
        "phase3_validation_scores.parquet",
        "phase3_metrics.csv",
        "phase3_detector_comparison.png",
        "provenance.json",
    }
    assert {path.name for path in config.output_dir.iterdir()} == expected
    assert metrics_path == config.output_dir / "phase3_metrics.csv"
    provenance = json.loads((config.output_dir / "provenance.json").read_text())
    assert provenance["frozen"] is True
    assert provenance["device"] == "cpu"

    mean = load_detector(config.output_dir / "mean_detector.pt")
    logistic = load_detector(config.output_dir / "logistic_detector.pt")
    assert mean["selected_layer"] == logistic["selected_layer"] == 2
    assert mean["phase1_run_id"] == logistic["phase1_run_id"] == "phase1-run"
    assert logistic["settings"]["solver"] == "liblinear"
    np.testing.assert_array_equal(logistic["feature_token_ids"].numpy(), [10, 20])

    scores = pd.read_parquet(config.output_dir / "phase3_validation_scores.parquet")
    train = np.array([0, 1, 2, 3])
    validation = np.array([4, 5, 6, 7])
    features, _ = build_sparse_features(
        handoff, train, validation, logistic["feature_token_ids"].numpy()
    )
    expected_logits = features @ logistic["weights"].numpy() + logistic["intercept"]
    np.testing.assert_allclose(scores.logistic_score, expected_logits)
    assert list(scores.columns) == [
        "example_id",
        "example_index",
        "pair_id",
        "task",
        "task_display",
        "condition",
        "label",
        "mean_score",
        "logistic_score",
        "mean_prediction",
        "logistic_prediction",
    ]

    metrics = pd.read_csv(metrics_path)
    assert set(metrics.detector) == {"mean", "logistic"}
    assert set(metrics.metric) == {
        "auprc",
        "auroc",
        "balanced_accuracy",
        "tpr",
        "fpr",
    }
    assert len(metrics) == 16
