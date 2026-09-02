from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml
from sklearn.metrics import average_precision_score

from jspace_research.phase1.config import EXPECTED_BIPIA_REVISION
from jspace_research.phase2.scoring import JUDGE_RUBRIC_SHA256
from jspace_research.phase4.agentdojo import (
    _contains_injection,
    _native_cases,
    _normalize_gemma_tool_call,
    validate_smoke_records,
)
from jspace_research.phase4.bipia import (
    MANIFEST_NAME,
    TEST_VARIANTS,
    _build_task_cases,
    _validate_manifest,
    load_manifest,
)
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
            "task": "email",
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
            "task": "email",
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
    assert {"overall", "task", "macro"}.issubset(
        set(metrics[metrics.benchmark == "bipia"].scope)
    )
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


@pytest.mark.parametrize(
    ("completion", "expected_call"),
    [
        (
            'thought\ncall:read_file{"file_path": "bill-december-2023.txt"}',
            '<function=read_file>{"file_path":"bill-december-2023.txt"}</function>',
        ),
        (
            "thought\ncall:get_most_recent_transactions{n:100}",
            '<function=get_most_recent_transactions>{"n":100}</function>',
        ),
        (
            'thought\n<function=get_webpage={"url": "www.informations.com"}></function>',
            '<function=get_webpage>{"url":"www.informations.com"}</function>',
        ),
        (
            'call:search_calendar_events{query: "Networking event",date: "2025-05-26"}',
            '<function=search_calendar_events>{"query":"Networking event","date":"2025-05-26"}</function>',
        ),
    ],
)
def test_agentdojo_normalizes_observed_gemma_tool_calls(
    completion: str, expected_call: str
) -> None:
    assert expected_call in _normalize_gemma_tool_call(completion)


def test_agentdojo_leaves_unparseable_tool_calls_for_native_parser() -> None:
    completion = "call:read_file{not valid arguments}"
    assert _normalize_gemma_tool_call(completion) == completion


def test_agentdojo_finds_injection_after_native_yaml_formatting() -> None:
    injection = "<INFORMATION>\n\nImportant instruction\n\nwith a second line\n</INFORMATION>"
    collapsed = "<INFORMATION>\nImportant instruction\nwith a second line\n</INFORMATION>"
    tool_text = yaml.safe_dump({"subject": collapsed}, width=20)
    assert injection not in tool_text
    assert _contains_injection([tool_text], [injection]) is True
    assert _contains_injection([tool_text], ["unrelated instruction"]) is False


def test_agentdojo_smoke_requires_scored_clean_and_exposed_attack_per_suite() -> None:
    records = [
        {
            "subgroup": "banking",
            "condition": "control",
            "injection_exposed": False,
            "mean_score": -1.0,
        },
        {
            "subgroup": "banking",
            "condition": "attack",
            "injection_exposed": True,
            "mean_score": 1.0,
        },
    ]
    validate_smoke_records(records, ["banking"])
    records[1]["injection_exposed"] = False
    records[1]["mean_score"] = None
    with pytest.raises(RuntimeError, match="suite banking"):
        validate_smoke_records(records, ["banking"])


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


class _FakeBipiaBuilder:
    def construct_prompt(self, example, require_system_prompt=True):
        assert require_system_prompt is True
        return "system", f"{example['question']}\n{example['context']}"

    def construct_response(self, record):
        return record["ideal"]


def _insert(label):
    def apply(context, attack, random_state=None):
        return f"{label}:{context}:{attack}:{random_state}"

    return apply


def _manifest_fixture(tmp_path):
    records = [
        {
            "context_id": f"email:test:{index:05d}",
            "context": f"context {index}",
            "question": f"question {index}",
            "ideal": f"answer {index}",
        }
        for index in range(100)
    ]
    attacks = {
        f"category-{category}": [f"attack {category}-{variant}" for variant in TEST_VARIANTS]
        for category in range(15)
    }
    insertions = {position: _insert(position) for position in ("start", "middle", "end")}
    cases, details = _build_task_cases(
        task="email",
        records=records,
        attacks=attacks,
        builder=_FakeBipiaBuilder(),
        insertions=insertions,
        attack_count=250,
        seed=42,
    )
    metadata = {
        "record_type": "metadata",
        "schema_version": 1,
        "bipia_revision": EXPECTED_BIPIA_REVISION,
        "seed": 42,
        "mode": "scientific",
        "attacks_per_task": 250,
        "positions": ["start", "middle", "end"],
        "test_variants": list(TEST_VARIANTS),
        "tasks": {"email": details},
    }
    config = SimpleNamespace(
        output_dir=tmp_path,
        smoke=False,
        phase1=SimpleNamespace(
            seed=42,
            tasks=("email",),
            dependencies=SimpleNamespace(bipia_revision=EXPECTED_BIPIA_REVISION),
        ),
    )
    return config, [metadata, *cases]


def test_bipia_manifest_is_deterministic_balanced_and_context_matched(tmp_path) -> None:
    config, rows = _manifest_fixture(tmp_path)
    cases = _validate_manifest(config, rows)
    repeated_config, repeated_rows = _manifest_fixture(tmp_path)
    assert repeated_config.phase1.seed == config.phase1.seed
    assert repeated_rows == rows

    attacks = [row for row in cases if row["condition"] == "attack"]
    controls = [row for row in cases if row["condition"] == "control"]
    assert len(attacks) == 250
    assert len(controls) == 100
    assert Counter(row["attack_variant_id"] for row in attacks) == Counter(
        {variant: 50 for variant in TEST_VARIANTS}
    )
    cell_counts = Counter((row["attack_category"], row["position"]) for row in attacks)
    assert max(cell_counts.values()) - min(cell_counts.values()) == 1
    context_counts = Counter(row["context_id"] for row in attacks)
    assert max(context_counts.values()) - min(context_counts.values()) == 1
    assert len({row["prompt_hash"] for row in attacks}) == 250
    assert {row["source_clean_case_id"] for row in attacks} == {
        row["case_id"] for row in controls
    }


def test_bipia_manifest_freezing_rejects_changed_case(tmp_path) -> None:
    config, rows = _manifest_fixture(tmp_path)
    path = tmp_path / MANIFEST_NAME
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert len(load_manifest(config)) == 350

    changed = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    changed["prompt_hash"] = "0" * 64
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = json.dumps(changed)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="prompt changed"):
        load_manifest(config)


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
    assert "RESEARCH_REVISION = 'main'" in install_cell
    assert "DATA_ROOT = Path('/content/drive/MyDrive/jspace-research/data')" in config_cell
    assert "if RUN_MODE == 'full':" in config_cell
    assert "DATA_ROOT / 'webqa/train.jsonl'" in config_cell
    assert "DATA_ROOT / 'summarization/train.jsonl'" in config_cell
    assert "shutil.copy2(webqa_test" in config_cell
    assert "shutil.copy2(summarization_test" in config_cell
