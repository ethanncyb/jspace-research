# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
import transformers

import jlens.study as study_module
from jlens.local import compute_local_jacobian, run_local_steering_sweep
from jlens.study import (
    EXPECTED_GSM8K_CASES,
    EXPECTED_HUMANEVAL_CASES,
    MODEL_REGISTRY,
    BehaviorCase,
    JsonlStore,
    RunManifest,
    StudyConfig,
    build_local_jspace_page,
    case_seed,
    extract_numeric_answer,
    extract_python_code,
    load_quantized_model,
    score_gsm8k,
    score_humaneval,
    split_thinking,
    validate_gpu,
)

from .tiny import TinyDecoder


def test_expected_full_benchmark_cardinalities() -> None:
    assert EXPECTED_HUMANEVAL_CASES == 164
    assert EXPECTED_GSM8K_CASES == 1_319


def test_registry_and_config_hash_are_stable_across_output_locations(tmp_path):
    assert list(MODEL_REGISTRY) == [
        "qwen3-4b",
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-32b",
    ]
    first = StudyConfig("qwen3-4b", output_root=str(tmp_path / "a"))
    second = replace(first, output_root=str(tmp_path / "b"))
    assert first.config_hash == second.config_hash
    assert first.experiment_hash == second.experiment_hash
    assert first.experiment_hash == StudyConfig("qwen3-8b").experiment_hash
    assert first.model_revision == MODEL_REGISTRY["qwen3-4b"].revision
    assert StudyConfig("qwen3-8b").model_revision == MODEL_REGISTRY["qwen3-8b"].revision
    assert first.humaneval_revision == "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
    assert first.gsm8k_revision == "740312add88f781978c0658806c59bc2815b9866"
    assert case_seed(17, "x") == case_seed(17, "x")
    assert case_seed(17, "x") != case_seed(17, "y")


def test_gpu_validation_enforces_32b_a100_and_vram():
    with pytest.raises(RuntimeError, match="A100"):
        validate_gpu(
            MODEL_REGISTRY["qwen3-32b"],
            {"cuda_available": True, "bf16_supported": True, "vram_gb": 48, "gpu": "L40"},
        )
    with pytest.raises(RuntimeError, match="at least"):
        validate_gpu(
            MODEL_REGISTRY["qwen3-4b"],
            {"cuda_available": True, "bf16_supported": True, "vram_gb": 16, "gpu": "L4"},
        )
    assert validate_gpu(
        MODEL_REGISTRY["qwen3-32b"],
        {"cuda_available": True, "bf16_supported": True, "vram_gb": 39.5, "gpu": "NVIDIA A100-SXM4-40GB"},
    ) == []


def test_quantized_loader_uses_frozen_nf4_bf16_contract(monkeypatch):
    class DummyTokenizer:
        pad_token_id = None
        eos_token_id = 1

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    tokenizer = DummyTokenizer()
    hf_model = DummyModel()
    captured = {}

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    def fake_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        return hf_model

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(study_module, "from_hf", lambda model, tok: "adapter")

    loaded, loaded_tokenizer, adapter = load_quantized_model(
        StudyConfig("qwen3-4b")
    )

    quantization = captured["quantization_config"]
    assert quantization.load_in_4bit
    assert quantization.bnb_4bit_quant_type == "nf4"
    assert quantization.bnb_4bit_compute_dtype == torch.bfloat16
    assert quantization.bnb_4bit_use_double_quant
    assert captured["dtype"] == torch.bfloat16
    assert captured["device_map"] == {"": 0}
    assert captured["attn_implementation"] == "sdpa"
    assert loaded is hf_model and loaded_tokenizer is tokenizer
    assert adapter == "adapter" and tokenizer.pad_token_id == 1
    assert not hf_model.weight.requires_grad


def test_jsonl_store_resumes_and_rejects_mismatched_metadata(tmp_path):
    path = tmp_path / "rows.jsonl"
    store = JsonlStore(path, config_hash="abc")
    assert store.append({"case_id": "one", "value": 1})
    assert not store.append({"case_id": "one", "value": 2})
    resumed = JsonlStore(path, config_hash="abc")
    assert resumed.contains("one")
    assert resumed.rows() == [{"case_id": "one", "value": 1}]
    with pytest.raises(ValueError, match="metadata"):
        JsonlStore(path, config_hash="different")


def test_manifest_is_idempotent_but_not_cross_config(tmp_path):
    path = tmp_path / "manifest.json"
    first = StudyConfig("qwen3-4b")
    RunManifest.create(first, environment={}).write(path)
    RunManifest.create(first, environment={}).write(path)
    payload = json.loads(path.read_text())
    assert payload["config_hash"] == first.config_hash
    with pytest.raises(ValueError, match="another configuration"):
        RunManifest.create(StudyConfig("qwen3-8b"), environment={}).write(path)


def test_thinking_code_and_numeric_parsers():
    thinking, answer = split_thinking("<think>work</think>```python\ndef f(): return 1\n```")
    assert thinking == "work"
    assert extract_python_code(answer) == "def f(): return 1"
    assert extract_numeric_answer("reasoning 8 then \\boxed{1,024}") == "1024"
    assert extract_numeric_answer("last values 2 and -3.5") == "-3.5"
    assert score_gsm8k("steps\n#### 24", "Therefore \\boxed{24}")["passed"]
    malformed = score_gsm8k("#### 24", "no numeric answer")
    assert not malformed["passed"] and not malformed["parsed"]


def test_humaneval_scoring_executes_in_isolated_subprocess():
    case = BehaviorCase(
        "humaneval",
        "tiny",
        "def add(a, b):\n    ",
        "return a + b",
        "def check(candidate):\n    assert candidate(2, 3) == 5",
        "add",
    )
    assert score_humaneval(case, "```python\ndef add(a, b):\n    return a + b\n```")["passed"]
    failure = score_humaneval(case, "return a - b")
    assert not failure["passed"]


def test_heatmap_contains_sensitivity_and_random_adjusted_panels():
    model = TinyDecoder()
    local = compute_local_jacobian(
        model, "abc", target_token_id=5, layers=list(range(model.n_layers))
    )
    sweep = run_local_steering_sweep(
        model, local, layers=[1, 2], strengths=[0, 0.1]
    )
    page = build_local_jspace_page(
        local, [row.to_dict() for row in sweep], model.tokenizer
    )
    assert "Prompt-local target-logit sensitivity" in page
    assert "random-adjusted target-rank gain" in page
    assert "L1" in page and "L2" in page
