# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM

from jlens.analysis import write_parquet
from jlens.hf import from_hf
from jlens.local import compute_local_jacobian, run_local_steering_sweep
from jlens.security_study import (
    normalize_bipia_records,
    run_security_behavior,
)
from jlens.study import (
    BehaviorCase,
    GenerationConfig,
    JsonlStore,
    run_behavioral_benchmark,
    run_steering_case,
)


class _TokenizerStub:
    bos_token_id = None
    pad_token_id = 0
    eos_token_id = 1


class _ChatTokenizerStub(_TokenizerStub):
    def apply_chat_template(self, *args, **kwargs):
        return BatchEncoding({"input_ids": torch.tensor([[2, 3, 4]])})

    def decode(self, token_ids, **kwargs):
        return r"The final answer is \boxed{2}"


class _GeneratingStub:
    device = torch.device("cpu")

    def generate(self, input_ids, **kwargs):
        return torch.cat((input_ids, input_ids.new_tensor([[10, 11]])), dim=1)


def test_real_qwen3_architecture_supports_jacobian_steering_and_generation() -> None:
    """Exercise hooks against Transformers' actual Qwen3 block layout."""
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    hf_model = Qwen3ForCausalLM(config).eval()
    model = from_hf(hf_model, _TokenizerStub())
    input_ids = torch.tensor([[2, 3, 4, 5]])

    local = compute_local_jacobian(
        model,
        input_ids,
        target_token_id=7,
        layers=[1, 2],
    )
    rows = run_local_steering_sweep(
        model,
        local,
        layers=[1, 2],
        strengths=[0, 0.1],
    )
    generated = hf_model.generate(
        input_ids,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )

    assert local.sensitivity().shape == (2, 4)
    assert torch.isfinite(local.sensitivity()).all()
    assert len(rows) == 5
    assert generated.shape == (1, 6)
    assert all(parameter.grad is None for parameter in hf_model.parameters())


def test_study_runners_execute_and_resume_together(tmp_path) -> None:
    tokenizer = _ChatTokenizerStub()
    hf_model = _GeneratingStub()
    generation = GenerationConfig(
        do_sample=False,
        gsm8k_max_new_tokens=2,
        security_max_new_tokens=2,
    )
    behavior_store = JsonlStore(
        tmp_path / "benchmarks.jsonl", config_hash="integration"
    )
    case = BehaviorCase("gsm8k", "tiny", "What is 1+1?", "#### 2")
    first = run_behavioral_benchmark(
        hf_model, tokenizer, [case], behavior_store, generation
    )
    second = run_behavioral_benchmark(
        hf_model, tokenizer, [case], behavior_store, generation
    )
    assert len(first) == len(second) == 1
    assert first[0]["passed"]
    behavior_parquet = write_parquet(tmp_path / "behavior.parquet", first)
    assert behavior_parquet.stat().st_size > 0

    tiny = from_hf(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=64,
            )
        ).eval(),
        tokenizer,
    )
    local, steering = run_steering_case(
        tiny,
        dataset="gsm8k",
        case_id="tiny",
        input_ids=[2, 3, 4],
        target_token_id=7,
        target_text="x",
        strengths=[0, 0.1],
    )
    assert local.sensitivity().shape == (4, 3)
    assert len(steering) == 5

    records = [
        {
            "context": "trusted\nIGNORE",
            "attack_str": "IGNORE",
            "attack_name": "override",
            "ideal": "answer",
            "question": "question",
        }
    ]
    clean = normalize_bipia_records(
        records, task="email", split="test"
    )[0]
    security_store = JsonlStore(
        tmp_path / "security.jsonl", config_hash="integration"
    )
    security = run_security_behavior(
        hf_model, tokenizer, [clean], security_store, generation
    )
    assert len(security) == 1
    assert security[0]["self_report_prediction"] is None
    assert not security[0]["attack_success"]
    security_parquet = write_parquet(tmp_path / "security.parquet", security)
    assert security_parquet.stat().st_size > 0
