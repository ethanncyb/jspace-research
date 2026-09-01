from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score

from jspace_research.phase2.scoring import JUDGE_RUBRIC_SHA256
from jspace_research.phase4.agentdojo import _native_cases
from jspace_research.phase4.common import (
    completed_records,
    content_hash,
    require_generation_context,
    save_record,
)
from jspace_research.phase4.detectors import FrozenDetectors
from jspace_research.phase4.injecagent import build_cases as build_injecagent_cases
from jspace_research.phase4.pipeline import _balanced_bipia_rows, _judge_bipia, _metrics


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


def test_phase4_uses_native_model_context_without_truncation() -> None:
    require_generation_context(5000, 262144, 512, "AgentDojo")
    with pytest.raises(RuntimeError, match="pinned model context window"):
        require_generation_context(262000, 262144, 512, "AgentDojo")


def test_phase4_uses_benchmark_specific_metrics() -> None:
    rows = [
        {
            "case_id": "bipia:email:test:00000:control",
            "benchmark": "bipia",
            "context_id": "email:test:00000",
            "condition": "control",
            "subgroup": "EmailQA",
            "injection_exposed": False,
            "mean_score": -1.0,
            "mean_prediction": False,
            "logistic_score": -1.0,
            "logistic_prediction": False,
            "native_utility": None,
            "native_attack_success": None,
            "native_valid": None,
        },
        {
            "case_id": "bipia:email:test:00000:attack:category:0:start",
            "benchmark": "bipia",
            "context_id": "email:test:00000",
            "condition": "attack",
            "subgroup": "EmailQA",
            "injection_exposed": True,
            "mean_score": 1.0,
            "mean_prediction": True,
            "logistic_score": 1.0,
            "logistic_prediction": True,
            "native_utility": None,
            "native_attack_success": None,
            "native_valid": None,
        },
        {
            "benchmark": "agentdojo",
            "condition": "control",
            "subgroup": "banking",
            "injection_exposed": False,
            "mean_score": -1.0,
            "mean_prediction": False,
            "logistic_score": -1.0,
            "logistic_prediction": False,
            "native_utility": True,
            "native_attack_success": None,
            "native_valid": None,
        },
        {
            "benchmark": "agentdojo",
            "condition": "attack",
            "subgroup": "banking",
            "injection_exposed": True,
            "mean_score": 1.0,
            "mean_prediction": True,
            "logistic_score": 1.0,
            "logistic_prediction": True,
            "native_utility": False,
            "native_attack_success": True,
            "native_valid": None,
        },
        {
            "benchmark": "agentdojo",
            "condition": "control",
            "subgroup": "slack",
            "injection_exposed": False,
            "mean_score": None,
            "mean_prediction": None,
            "logistic_score": None,
            "logistic_prediction": None,
            "native_utility": True,
            "native_attack_success": None,
            "native_valid": None,
        },
        {
            "benchmark": "agentdojo",
            "condition": "attack",
            "subgroup": "slack",
            "injection_exposed": False,
            "mean_score": None,
            "mean_prediction": None,
            "logistic_score": None,
            "logistic_prediction": None,
            "native_utility": False,
            "native_attack_success": False,
            "native_valid": None,
        },
        {
            "benchmark": "injecagent",
            "condition": "attack",
            "subgroup": "direct_harm",
            "injection_exposed": True,
            "mean_score": 1.0,
            "mean_prediction": True,
            "logistic_score": 1.0,
            "logistic_prediction": True,
            "native_utility": None,
            "native_attack_success": True,
            "native_valid": True,
        },
        {
            "benchmark": "injecagent",
            "condition": "attack",
            "subgroup": "data_stealing",
            "injection_exposed": True,
            "mean_score": -1.0,
            "mean_prediction": False,
            "logistic_score": -1.0,
            "logistic_prediction": False,
            "native_utility": None,
            "native_attack_success": False,
            "native_valid": False,
        },
    ]
    metrics = _metrics(pd.DataFrame(rows), detectors())
    assert set(metrics[metrics.benchmark == "bipia"].metric) == {
        "auprc",
        "auroc",
        "tpr",
        "fpr",
        "balanced_accuracy",
    }
    assert "auprc" not in set(metrics[metrics.benchmark == "injecagent"].metric)
    assert {"valid_rate", "asr_valid", "asr_all"}.issubset(
        set(metrics[metrics.benchmark == "injecagent"].metric)
    )
    undefined_tpr = metrics[
        (metrics.benchmark == "agentdojo")
        & (metrics.subgroup == "slack")
        & (metrics.metric == "tpr")
        & (metrics.detector == "mean")
    ].iloc[0]
    assert undefined_tpr.n == 0
    assert pd.isna(undefined_tpr.value)


def test_phase4_record_resumption_rejects_stale_identity(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    identity = {"phase4_config_sha256": "a" * 64}
    save_record(
        path,
        {
            **identity,
            "case_id": "case-1",
            "case_hash": "b" * 64,
            "benchmark": "bipia",
            "context_id": "email:test:00000",
            "condition": "attack",
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


def test_agentdojo_smoke_uses_first_sorted_native_cases() -> None:
    suite = SimpleNamespace(
        user_tasks={"user-2": object(), "user-1": object()},
        injection_tasks={"injection-2": object(), "injection-1": object()},
    )
    cases = _native_cases(suite, smoke=True)
    identities = [
        (
            condition,
            user[0],
            None if injection is None else injection[0],
        )
        for condition, user, injection in cases
    ]
    assert identities == [
        ("control", "user-1", None),
        ("control", "user-2", None),
        ("attack", "user-1", "injection-1"),
        ("attack", "user-1", "injection-2"),
    ]


def test_bipia_metrics_balance_clean_scores_by_source_context() -> None:
    rows = [
        {
            "case_id": "bipia:email:test:00000:control",
            "context_id": "email:test:00000",
            "benchmark": "bipia",
            "condition": "control",
            "mean_score": 0.0,
        },
        *[
            {
                "case_id": f"bipia:email:test:00000:attack:{index}",
                "context_id": "email:test:00000",
                "benchmark": "bipia",
                "condition": "attack",
                "mean_score": 0.0,
            }
            for index in range(3)
        ],
    ]
    balanced = _balanced_bipia_rows(pd.DataFrame(rows))
    assert len(balanced[balanced.condition == "attack"]) == 3
    assert len(balanced[balanced.condition == "control"]) == 3
    assert float((balanced.condition == "attack").mean()) == pytest.approx(0.5)
    labels = (balanced.condition == "attack").to_numpy(dtype=int)
    assert average_precision_score(labels, balanced.mean_score) == pytest.approx(0.5)


def test_injecagent_requires_the_frozen_1054_base_cases(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "test_cases_dh_base.json").write_text(json.dumps([{}] * 510), encoding="utf-8")
    (data / "test_cases_ds_base.json").write_text(json.dumps([{}] * 544), encoding="utf-8")
    config = SimpleNamespace(injecagent_root=tmp_path, smoke=False)
    assert len(build_injecagent_cases(config)) == 1054

    (data / "test_cases_dh_base.json").write_text(json.dumps([{}] * 509), encoding="utf-8")
    with pytest.raises(RuntimeError, match="must contain 510"):
        build_injecagent_cases(config)


def test_bipia_judgment_cache_rejects_inconsistent_outcome(tmp_path) -> None:
    config = SimpleNamespace(
        output_dir=tmp_path,
        judge_model="openai/gpt-4.1-mini",
        identity_hash=lambda: "a" * 64,
    )
    record = {
        "case_id": "bipia:case-1",
        "case_hash": "b" * 64,
        "condition": "attack",
        "generated_response": "response",
        "attack_text": "injection",
    }
    expected = {
        "schema_version": 1,
        "case_id": record["case_id"],
        "phase4_config_sha256": config.identity_hash(),
        "case_hash": record["case_hash"],
        "judge_gateway": "openrouter",
        "judge_model": config.judge_model,
        "judge_rubric_sha256": JUDGE_RUBRIC_SHA256,
        "generation_hash": content_hash(record["generated_response"]),
        "judge_label": "YES",
        "attack_success": False,
        "returned_model": None,
        "provider": None,
        "provider_model": None,
    }
    save_record(tmp_path / "bipia_judgments.jsonl", expected)
    with pytest.raises(RuntimeError, match="Inconsistent"):
        _judge_bipia(config, [record], judge=None)


def test_end_to_end_notebook_runs_phase4_after_phase3() -> None:
    notebook_path = Path(__file__).parents[1] / "notebooks/JSpace_End_to_End_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    phase3_index = next(index for index, text in enumerate(cells) if "phase3_command" in text)
    phase4_index = next(index for index, text in enumerate(cells) if "phase4_base" in text)
    persistence_index = next(
        index for index, text in enumerate(cells) if "Confirm persistence" in text
    )
    assert phase3_index < phase4_index < persistence_index
    assert "import pandas as pd" in cells[phase4_index]
    assert "from IPython.display import Image, display" in cells[phase4_index]
    assert "selection = json.loads" in cells[persistence_index + 1]

    config_cell = next(text for text in cells if "RUN_MODE = 'smoke'" in text)
    install_cell = next(text for text in cells if "RESEARCH_REVISION" in text)
    assert "RESEARCH_REVISION = 'remaining-phases'" in install_cell
    assert "DATA_ROOT = Path('/content/drive/MyDrive/jspace-research/data')" in config_cell
    assert "if RUN_MODE == 'full':" in config_cell
    assert "DATA_ROOT / 'webqa/train.jsonl'" in config_cell
    assert "DATA_ROOT / 'summarization/train.jsonl'" in config_cell
    assert "shutil.copy2(webqa_test" in config_cell
    assert "shutil.copy2(summarization_test" in config_cell
