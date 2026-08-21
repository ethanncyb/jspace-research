from __future__ import annotations

from pathlib import Path

import torch

from gsm8k_jspace.capture.hooks import JSpaceCapture
from gsm8k_jspace.config import (
    CaptureFields,
    CaptureSection,
    ConfigError,
    TokenSelector,
    config_from_mapping,
    load_config,
)
from gsm8k_jspace.models.jlens_adapter import JLensAdapter


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _identity_adapter(model):
    import torch as torch_mod
    import jlens

    eye = torch_mod.eye(model.d_model)
    lens = jlens.JacobianLens(
        {layer: eye.clone() for layer in range(model.n_layers)},
        n_prompts=0,
        d_model=model.d_model,
    )
    return JLensAdapter(lens, source_desc="identity", placeholder=True)


def test_capture_does_not_modify_output(tiny_model):
    adapter = _identity_adapter(tiny_model)
    cfg = CaptureSection(tokens=TokenSelector(mode="prompt_last"))
    ids = tiny_model.encode("ab")
    with JSpaceCapture(
        tiny_model,
        adapter,
        layers=[1],
        capture_cfg=cfg,
        prompt_len=ids.shape[1],
        run_id="r",
        example_id="e",
        condition="baseline",
    ) as capture:
        out = tiny_model.forward(ids)
    residual = out.last_hidden_state
    with torch.no_grad():
        clean = tiny_model.forward(ids).last_hidden_state
    assert torch.equal(residual, clean)
    assert capture.records
    for rec in capture.records:
        assert rec["layer"] == 1


def test_full_sequence_replay_records_every_token(tiny_model):
    adapter = _identity_adapter(tiny_model)
    cfg = CaptureSection(
        tokens=TokenSelector(mode="full_sequence", include_prompt=True),
        top_k_tokens=3,
    )
    ids = tiny_model.encode("ab")
    capture = JSpaceCapture(
        tiny_model,
        adapter,
        layers=[1, 2],
        capture_cfg=cfg,
        prompt_len=1,
        run_id="r",
        example_id="e",
        condition="baseline",
    )
    capture.capture_sequence_replay(ids, tiny_model.tokenizer)
    seq = int(ids.shape[1])
    layers = {row["layer"] for row in capture.records}
    assert layers == {1, 2}
    assert len(capture.records) == seq * 2
    assert all(row["capture_event"] == "sequence_replay" for row in capture.records)
    assert all(row.get("top_jspace_tokens") for row in capture.records)
    assert all(row.get("top_logit_tokens") for row in capture.records)
    assert all(len(row["top_jspace_tokens"]) == 3 for row in capture.records)
    last_layer_rows = [row for row in capture.records if row["layer"] == 2]
    assert all(row.get("top_model_tokens") for row in last_layer_rows)


def test_live_capture_all_layers_during_inference(tiny_model):
    adapter = _identity_adapter(tiny_model)
    layers = list(range(tiny_model.n_layers))
    cfg = CaptureSection(
        tokens=TokenSelector(mode="all_generated", include_prompt=False),
        top_k_tokens=3,
        fields=CaptureFields(
            hidden_norm=True,
            jspace_norm=True,
            top_jspace_tokens=True,
            top_logit_tokens=True,
            top_model_tokens=True,
        ),
    )
    prompt = tiny_model.encode("ab")
    prompt_len = int(prompt.shape[1])
    generated: list[int] = []
    with JSpaceCapture(
        tiny_model,
        adapter,
        layers=layers,
        capture_cfg=cfg,
        prompt_len=prompt_len,
        run_id="r",
        example_id="e",
        condition="baseline",
    ) as capture:
        current = prompt
        for _ in range(4):
            hidden = tiny_model.forward(current).last_hidden_state
            token_id = int(tiny_model.unembed(hidden[:, -1])[0].argmax().item())
            generated.append(token_id)
            current = torch.cat(
                [current, torch.tensor([[token_id]], dtype=current.dtype)], dim=1
            )
    capture.attach_tokens(generated, tiny_model.tokenizer)
    assert capture.records
    assert {row["layer"] for row in capture.records} == set(layers)
    assert all(row["capture_event"] == "decode" for row in capture.records)
    assert all(row.get("top_jspace_tokens") for row in capture.records)
    assert all(row.get("top_logit_tokens") for row in capture.records)
    assert all(len(row["top_jspace_tokens"]) == 3 for row in capture.records)
    last = max(layers)
    assert all(
        row.get("top_model_tokens")
        for row in capture.records
        if row["layer"] == last
    )
    by_layer = {layer: 0 for layer in layers}
    for row in capture.records:
        by_layer[int(row["layer"])] += 1
    assert len(set(by_layer.values())) == 1
    assert by_layer[0] == len(generated) - 1


def test_vector_sidecar_written_when_enabled(tiny_model, tmp_path: Path):
    adapter = _identity_adapter(tiny_model)
    cfg = CaptureSection(
        tokens=TokenSelector(mode="full_sequence", include_prompt=True),
        top_k_tokens=2,
        fields=CaptureFields(
            hidden_norm=True,
            jspace_norm=True,
            top_jspace_tokens=False,
            top_logit_tokens=False,
            top_model_tokens=False,
            hidden_vector=True,
            jspace_vector=True,
            intervention_delta_norm=False,
        ),
        vector_dtype="float16",
    )
    ids = tiny_model.encode("ab")
    capture = JSpaceCapture(
        tiny_model,
        adapter,
        layers=[1],
        capture_cfg=cfg,
        prompt_len=1,
        run_id="r",
        example_id="vec_ex",
        condition="baseline",
    )
    capture.capture_sequence_replay(ids, tiny_model.tokenizer)
    meta = capture.save(tmp_path / "vec_ex.jsonl")
    assert meta["vectors_path"] == "vec_ex.vectors.pt"
    vectors = torch.load(tmp_path / "vec_ex.vectors.pt", weights_only=True)
    assert vectors
    for row in capture.records:
        href = row["hidden_vector_ref"]
        jref = row["jspace_vector_ref"]
        assert href["key"] in vectors
        assert jref["key"] in vectors
        assert vectors[href["key"]].dtype == torch.float16


def test_vector_refs_omitted_when_disabled(tiny_model, tmp_path: Path):
    adapter = _identity_adapter(tiny_model)
    cfg = CaptureSection(
        tokens=TokenSelector(mode="full_sequence", include_prompt=True),
        fields=CaptureFields(hidden_vector=False, jspace_vector=False),
    )
    ids = tiny_model.encode("ab")
    capture = JSpaceCapture(
        tiny_model,
        adapter,
        layers=[1],
        capture_cfg=cfg,
        prompt_len=1,
        run_id="r",
        example_id="novec",
        condition="baseline",
    )
    capture.capture_sequence_replay(ids, tiny_model.tokenizer)
    meta = capture.save(tmp_path / "novec.jsonl")
    assert "vectors_path" not in meta
    assert all("hidden_vector_ref" not in row for row in capture.records)


def test_top_k_tokens_must_be_positive():
    data = load_config(CONFIGS / "default.yaml").to_dict()
    data["capture"]["top_k_tokens"] = 0
    try:
        config_from_mapping(data)
        raise AssertionError("expected ConfigError")
    except ConfigError as exc:
        assert "top_k_tokens" in str(exc)


def test_write_progress_updates_progress_json(tmp_path: Path):
    from gsm8k_jspace.artifacts.writer import write_json
    from gsm8k_jspace.artifacts.manifest import write_progress

    write_json(
        tmp_path / "manifest.json",
        {"status": "running", "completed_examples": 0, "total_examples": 5},
    )
    write_progress(
        tmp_path,
        completed_examples=2,
        total_examples=5,
        status="running",
        last_example_id="gsm8k_test_000001",
    )
    progress = (tmp_path / "progress.json").read_text()
    assert '"completed_examples": 2' in progress
    assert "gsm8k_test_000001" in progress


def test_capture_disabled_installs_no_hooks(tiny_model):
    assert tiny_model.layers[0]._forward_hooks == {}
