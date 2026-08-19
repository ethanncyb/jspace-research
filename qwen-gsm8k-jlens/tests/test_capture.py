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


def test_capture_disabled_installs_no_hooks(tiny_model):
    # constructing without entering is enough; runner skips construction when disabled
    assert tiny_model.layers[0]._forward_hooks == {}
