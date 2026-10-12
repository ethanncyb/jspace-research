import pytest
import torch
import torch.nn as nn
from jspace_research.analysis.stronger_ablation import (
    DirectionAblationHook,
    generate_random_directions,
)


def test_direction_ablation_hook_prefill_and_decoding():
    hidden_dim = 16
    layers = [0]
    unit_dir = torch.randn(hidden_dim)
    unit_dir = unit_dir / unit_dir.norm()
    directions = {0: unit_dir}

    hook = DirectionAblationHook(
        layers=layers,
        directions=directions,
        output_window=2,
        alpha=1.0,
    )

    class DummyLayer(nn.Module):
        def forward(self, x):
            return x

    model_layer = DummyLayer()
    hook.register({0: model_layer})

    # Test prefill (seq_len > 1)
    prompt_tensor = torch.randn(1, 5, hidden_dim)
    # Add a known component along unit_dir
    prompt_tensor += 3.0 * unit_dir.view(1, 1, hidden_dim)
    out_prefill = model_layer(prompt_tensor)

    # After prefill hook, projection onto unit_dir should be ~0
    proj = torch.einsum("btd,d->bt", out_prefill, unit_dir)
    assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-5)
    assert hook.prefill_edit_norms[0] > 0.0

    # Test decoding step 1 (seq_len == 1)
    gen_tensor_1 = torch.randn(1, 1, hidden_dim) + 2.0 * unit_dir.view(1, 1, hidden_dim)
    out_gen_1 = model_layer(gen_tensor_1)
    proj_gen_1 = (out_gen_1[0, -1] * unit_dir).sum()
    assert torch.allclose(proj_gen_1, torch.tensor(0.0), atol=1e-5)
    assert len(hook.gen_edit_norms[0]) == 1

    # Test decoding step 2 (seq_len == 1)
    gen_tensor_2 = torch.randn(1, 1, hidden_dim) + 1.5 * unit_dir.view(1, 1, hidden_dim)
    out_gen_2 = model_layer(gen_tensor_2)
    proj_gen_2 = (out_gen_2[0, -1] * unit_dir).sum()
    assert torch.allclose(proj_gen_2, torch.tensor(0.0), atol=1e-5)
    assert len(hook.gen_edit_norms[0]) == 2

    # Test decoding step 3 (beyond output_window=2, should NOT be modified)
    gen_tensor_3 = torch.randn(1, 1, hidden_dim) + 5.0 * unit_dir.view(1, 1, hidden_dim)
    out_gen_3 = model_layer(gen_tensor_3)
    proj_gen_3 = (out_gen_3[0, -1] * unit_dir).sum()
    assert not torch.allclose(proj_gen_3, torch.tensor(0.0), atol=1e-2)
    assert len(hook.gen_edit_norms[0]) == 2

    # Clean up
    hook.remove()
    assert len(hook.handles) == 0


def test_generate_random_directions():
    layers = [14, 15, 16]
    hidden_dim = 64
    dirs = generate_random_directions(layers, hidden_dim, seed=42, device=torch.device("cpu"))
    assert set(dirs.keys()) == set(layers)
    for l in layers:
        vec = dirs[l].to(torch.float32)
        assert vec.shape == (hidden_dim,)
        assert pytest.approx(vec.norm().item(), abs=1e-2) == 1.0
