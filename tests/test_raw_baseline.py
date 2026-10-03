from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from jspace_research.phase1.config import (
    DataConfig,
    DependencyConfig,
    LensConfig,
    ModelConfig,
    Phase1Config,
)
from jspace_research.phase1.jspace import tensor_to_bfloat16_bits
from jspace_research.raw_baseline.artifacts import RawActivationCache
from jspace_research.raw_baseline.config import RawBaselineConfig
from jspace_research.raw_baseline.detectors import RawDetectors
from jspace_research.raw_baseline.pipeline import (
    _bootstrap_deltas,
    _task_balanced_means,
    _validation_metric_values,
    fit,
)


def test_raw_cache_reads_only_the_frozen_layer_as_float32(tmp_path) -> None:
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    bits = tensor_to_bfloat16_bits(values)
    path = tmp_path / "raw.dat"
    path.write_bytes(bits.tobytes())
    memmap = np.memmap(path, dtype=np.uint16, mode="r", shape=(2, 2, 2))
    cache = RawActivationCache(
        handoff=SimpleNamespace(),
        residuals=memmap,
        selected_position=1,
        layers=(1, 2),
        width=2,
    )
    result = cache.read(np.array([0, 1], dtype=np.int64))
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[3.0, 4.0], [7.0, 8.0]])


def test_raw_mean_is_task_balanced_and_scoring_matches_formula() -> None:
    rows = pd.DataFrame(
        {
            "task": ["a", "a", "b", "b", "b", "b"],
            "label": [0, 1, 0, 0, 1, 1],
        }
    )
    activations = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 4.0], [0.0, 6.0]],
        dtype=np.float32,
    )
    clean, attack = _task_balanced_means(activations, rows, ("a", "b"))
    np.testing.assert_allclose(clean, [0.0, 1.5])
    np.testing.assert_allclose(attack, [1.0, 2.5])

    direction = attack - clean
    unit = direction / np.linalg.norm(direction)
    detectors = RawDetectors(
        mean={
            "raw_width": 2,
            "mu_clean": torch.from_numpy(clean),
            "d_unit": torch.from_numpy(unit),
            "threshold": 0.5,
        },
        logistic={
            "raw_width": 2,
            "weights": torch.tensor([1.0, -1.0]),
            "intercept": 0.25,
            "threshold": 0.0,
        },
    )
    result = detectors.score(torch.tensor([2.0, 1.5]))
    assert result["mean_score"] == pytest.approx(np.sqrt(2.0))
    assert result["logistic_score"] == pytest.approx(0.75)
    assert result["mean_prediction"] is True
    assert result["logistic_prediction"] is True


def _paired_validation() -> pd.DataFrame:
    rows = []
    for task in ("email", "qa"):
        for pair in range(4):
            for label, condition in ((0, "control"), (1, "attack")):
                score = -1.0 if label == 0 else 1.0
                rows.append(
                    {
                        "task": task,
                        "pair_id": f"{task}:{pair}",
                        "condition": condition,
                        "label": label,
                        "jspace_mean_score": score,
                        "jspace_mean_prediction": label == 1,
                        "jspace_mean_threshold": 0.0,
                        "raw_mean_score": score,
                        "raw_mean_prediction": label == 1,
                        "raw_mean_threshold": 0.0,
                        "jspace_logistic_score": score,
                        "jspace_logistic_prediction": label == 1,
                        "jspace_logistic_threshold": 0.0,
                        "raw_logistic_score": score,
                        "raw_logistic_prediction": label == 1,
                        "raw_logistic_threshold": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    frame = _paired_validation()
    first = _bootstrap_deltas(
        frame,
        strata=["task"],
        unit="pair_id",
        metric_fn=_validation_metric_values,
        replicates=20,
        seed=42,
    )
    second = _bootstrap_deltas(
        frame,
        strata=["task"],
        unit="pair_id",
        metric_fn=_validation_metric_values,
        replicates=20,
        seed=42,
    )
    assert first == second
    assert all(interval == pytest.approx((0.0, 0.0)) for interval in first.values())


def _raw_config(tmp_path: Path) -> RawBaselineConfig:
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
    return RawBaselineConfig(
        phase1=phase1,
        phase1_selected_path=tmp_path / "phase1/selected_layer.json",
        jspace_phase3_dir=tmp_path / "phase3",
        jspace_phase4_dir=tmp_path / "phase4",
        agentdojo_root=tmp_path / "agentdojo",
        injecagent_root=tmp_path / "InjecAgent",
        output_dir=tmp_path / "raw_baseline",
        agentdojo_revision="089ed468cf3ed0322acc66b0211f26d9d90dbf60",
        agentdojo_version="v1.2.2",
        agentdojo_suites=("banking", "slack", "travel", "workspace"),
        agentdojo_attack="important_instructions",
        agentdojo_defense=None,
        injecagent_revision="f19c9f2c79a41046eb13c03c51a24c567a8ffa07",
        injecagent_setting="base",
        injecagent_prompt_type="InjecAgent",
        max_new_tokens=512,
        penalty="l2",
        regularization_c=1.0,
        solver="liblinear",
        fit_intercept=True,
        class_weight=None,
        random_state=42,
        max_iter=1000,
        tol=1e-4,
    )


def test_raw_fit_cpu_flow_writes_only_baseline_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _raw_config(tmp_path)
    config.jspace_phase3_dir.mkdir()
    (config.jspace_phase3_dir / "mean_detector.pt").write_bytes(b"mean")
    (config.jspace_phase3_dir / "logistic_detector.pt").write_bytes(b"logistic")
    examples = []
    vectors = []
    for split in ("train", "validation"):
        for pair in range(2):
            for condition, label in (("control", 0), ("attack", 1)):
                examples.append(
                    {
                        "example_index": len(examples),
                        "pair_id": f"email:{split}:{pair}",
                        "task": "email",
                        "task_display": "EmailQA",
                        "split": split,
                        "condition": condition,
                        "label": label,
                    }
                )
                vectors.append([float(label * 2 - 1), float(pair) / 10])
    bits = tensor_to_bfloat16_bits(torch.tensor(vectors, dtype=torch.float32)).reshape(
        len(vectors), 1, 2
    )
    handoff = SimpleNamespace(
        metadata={
            "run_id": "phase1-test",
            "config_sha256": config.phase1.identity_hash(),
            "manifest_sha256": "d" * 64,
            "selected_layer": 2,
        },
        selected_layer=2,
        examples=examples,
    )
    cache = RawActivationCache(
        handoff=handoff,
        residuals=bits,
        selected_position=0,
        layers=(2,),
        width=2,
    )
    from jspace_research.raw_baseline import pipeline

    monkeypatch.setattr(pipeline, "load_raw_activation_cache", lambda _: cache)
    monkeypatch.setattr(
        pipeline,
        "_reference_provenance",
        lambda _: {
            "run_id": "phase4-test",
            "phase1_run_id": "phase1-test",
            "phase4_config_sha256": "p4",
            "bipia_test_manifest_sha256": "m" * 64,
        },
    )
    monkeypatch.setattr(pipeline.FrozenDetectors, "load", lambda *args: object())
    fit(config)

    expected = {
        "raw_mean_detector.pt",
        "raw_logistic_detector.pt",
        "raw_validation_scores.parquet",
        "raw_validation_metrics.csv",
        "provenance.json",
    }
    assert {path.name for path in config.output_dir.iterdir()} == expected
    scores = pd.read_parquet(config.output_dir / "raw_validation_scores.parquet")
    assert bool(
        (scores[scores.label == 1].mean_score > scores[scores.label == 0].mean_score.max()).all()
    )
