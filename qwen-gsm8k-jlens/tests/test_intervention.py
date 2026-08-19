from __future__ import annotations

import torch

from gsm8k_jspace.config import FeatureSection, InterventionSection
from gsm8k_jspace.interventions.mean_replace import MeanReplaceController
from gsm8k_jspace.interventions.no_op import NoOpController
from gsm8k_jspace.models.jlens_adapter import JLensAdapter


def _adapter(model):
    import jlens

    eye = torch.eye(model.d_model)
    lens = jlens.JacobianLens(
        {layer: eye.clone() for layer in range(model.n_layers)},
        n_prompts=0,
        d_model=model.d_model,
    )
    return JLensAdapter(lens, source_desc="identity", placeholder=True)


def test_no_op_returns_same_object(tiny_model):
    ids = tiny_model.encode("abc")
    ctrl = NoOpController(tiny_model, layers=[1])
    with ctrl:
        out = tiny_model.layers[1](tiny_model.embed_tokens(ids))
    # After hook removal, compare a fresh forward through the same block
    hidden = tiny_model.embed_tokens(ids)
    with ctrl:
        first = tiny_model.layers[1](hidden)
        second = tiny_model.layers[1](hidden)
    assert first is not None
    assert torch.allclose(first, second)
    assert ctrl.summary()["delta_hidden_norm"] == 0.0


def test_strength_zero_is_exact(tiny_model):
    adapter = _adapter(tiny_model)
    spec = InterventionSection(method="mean_replace", strength=0.0, enabled=True)
    spec.features = FeatureSection(mode="top_abs", top_k=2)
    ctrl = MeanReplaceController(adapter, tiny_model, [1], spec)
    ids = tiny_model.encode("ab")
    hidden = tiny_model.embed_tokens(ids)
    with torch.no_grad():
        clean = tiny_model.layers[1](hidden).clone()
    with ctrl:
        out = tiny_model.layers[1](hidden)
    assert torch.equal(out, clean)


def test_reset_clears_running_mean(tiny_model):
    adapter = _adapter(tiny_model)
    spec = InterventionSection(method="mean_replace", strength=0.2, enabled=True)
    ctrl = MeanReplaceController(adapter, tiny_model, [1], spec)
    z = torch.ones(2, tiny_model.d_model)
    ctrl._update_running_mean(z, 1)
    assert ctrl._run_count[1] == 2
    ctrl.reset_example("e", 3)
    assert ctrl._run_count == {}
