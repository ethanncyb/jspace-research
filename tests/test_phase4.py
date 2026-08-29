from __future__ import annotations

import pandas as pd
import pytest
import torch

from jspace_research.phase4.common import completed_records, save_record
from jspace_research.phase4.detectors import FrozenDetectors
from jspace_research.phase4.pipeline import _metrics


def detectors() -> FrozenDetectors:
    decomposition = {"sparsity_k": 25, "screen_candidates": 512}
    return FrozenDetectors(
        mean={
            "selected_layer": 2,
            "decomposition": decomposition,
            "mu_clean": torch.zeros(3),
            "d_unit": torch.tensor([0.0, 0.0, 1.0]),
            "threshold": 0.5,
        },
        logistic={
            "selected_layer": 2,
            "decomposition": decomposition,
            "feature_token_ids": torch.tensor([0, 1]),
            "weights": torch.tensor([2.0, 3.0]),
            "intercept": -0.25,
            "threshold": 0.0,
        },
        dictionary=torch.eye(3),
    )


def test_frozen_scoring_does_not_expand_logistic_vocabulary() -> None:
    frozen = detectors()
    result = frozen.score(torch.tensor([0.0, 0.0, 1.0]), frozen.dictionary)
    assert result["mean_score"] == pytest.approx(1.0)
    assert result["mean_prediction"] is True
    assert result["logistic_score"] == pytest.approx(-0.25)
    assert result["logistic_prediction"] is False


def test_phase4_uses_benchmark_specific_metrics() -> None:
    rows = [
        {"benchmark": "bipia", "condition": "control", "subgroup": "EmailQA", "injection_exposed": False, "mean_score": -1.0, "mean_prediction": False, "logistic_score": -1.0, "logistic_prediction": False, "native_utility": None, "native_attack_success": None, "native_valid": None},
        {"benchmark": "bipia", "condition": "attack", "subgroup": "EmailQA", "injection_exposed": True, "mean_score": 1.0, "mean_prediction": True, "logistic_score": 1.0, "logistic_prediction": True, "native_utility": None, "native_attack_success": None, "native_valid": None},
        {"benchmark": "agentdojo", "condition": "control", "subgroup": "banking", "injection_exposed": False, "mean_score": -1.0, "mean_prediction": False, "logistic_score": -1.0, "logistic_prediction": False, "native_utility": True, "native_attack_success": None, "native_valid": None},
        {"benchmark": "agentdojo", "condition": "attack", "subgroup": "banking", "injection_exposed": True, "mean_score": 1.0, "mean_prediction": True, "logistic_score": 1.0, "logistic_prediction": True, "native_utility": False, "native_attack_success": True, "native_valid": None},
        {"benchmark": "injecagent", "condition": "attack", "subgroup": "direct_harm", "injection_exposed": True, "mean_score": 1.0, "mean_prediction": True, "logistic_score": 1.0, "logistic_prediction": True, "native_utility": None, "native_attack_success": True, "native_valid": True},
        {"benchmark": "injecagent", "condition": "attack", "subgroup": "data_stealing", "injection_exposed": True, "mean_score": -1.0, "mean_prediction": False, "logistic_score": -1.0, "logistic_prediction": False, "native_utility": None, "native_attack_success": False, "native_valid": False},
    ]
    metrics = _metrics(pd.DataFrame(rows), detectors())
    assert set(metrics[metrics.benchmark == "bipia"].metric) == {
        "auprc", "auroc", "tpr", "fpr", "balanced_accuracy"
    }
    assert "auprc" not in set(metrics[metrics.benchmark == "injecagent"].metric)
    assert {"valid_rate", "asr_valid", "asr_all"}.issubset(
        set(metrics[metrics.benchmark == "injecagent"].metric)
    )


def test_phase4_record_resumption_rejects_stale_identity(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    identity = {"phase4_config_sha256": "a" * 64}
    save_record(
        path,
        {
            **identity,
            "case_id": "case-1",
            "case_hash": "b" * 64,
            "generated_response": "response",
            "mean_score": 1.0,
            "mean_prediction": True,
            "logistic_score": 2.0,
            "logistic_prediction": True,
        },
    )
    assert set(completed_records(path, identity)) == {"case-1"}
    with pytest.raises(RuntimeError, match="identity mismatch"):
        completed_records(path, {"phase4_config_sha256": "c" * 64})
