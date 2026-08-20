from __future__ import annotations

import torch

from gsm8k_jspace.capture.hooks import JSpaceCapture
from gsm8k_jspace.config import CaptureSection, TokenSelector
from gsm8k_jspace.models.jlens_adapter import JLensAdapter


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
    last_layer_rows = [row for row in capture.records if row["layer"] == 2]
    assert all(row.get("top_model_tokens") for row in last_layer_rows)


def test_live_capture_all_layers_during_inference(tiny_model):
    adapter = _identity_adapter(tiny_model)
    layers = list(range(tiny_model.n_layers))
    cfg = CaptureSection(
        tokens=TokenSelector(mode="all_generated", include_prompt=False),
        top_k_tokens=3,
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
    by_layer = {layer: 0 for layer in layers}
    for row in capture.records:
        by_layer[int(row["layer"])] += 1
    assert len(set(by_layer.values())) == 1
    assert by_layer[0] == len(generated) - 1


def test_capture_disabled_installs_no_hooks(tiny_model):
    # constructing without entering is enough; runner skips construction when disabled
    assert tiny_model.layers[0]._forward_hooks == {}
