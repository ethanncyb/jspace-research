from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from jspace_research.phase1.jspace import tensor_to_bfloat16_bits
from jspace_research.phase1.robustness import (
    _comparison_layers,
    _load_or_compute_reconstruction,
    _plot_density_grid,
    _sample_indices,
)


def test_comparison_subset_is_deterministic_and_pair_complete() -> None:
    rows = []
    example_index = 0
    for split, pair_count in (("train", 220), ("validation", 120)):
        for task in ("email", "qa"):
            for pair in range(pair_count):
                for condition, label in (("control", 0), ("attack", 1)):
                    rows.append(
                        {
                            "example_index": example_index,
                            "pair_id": f"{split}-{task}-{pair}",
                            "split": split,
                            "task": task,
                            "condition": condition,
                            "label": label,
                        }
                    )
                    example_index += 1
    examples = pd.DataFrame(rows).set_index("example_index", drop=False)
    first = _sample_indices(examples, ("email", "qa"))
    second = _sample_indices(examples, ("email", "qa"))
    np.testing.assert_array_equal(first, second)
    selected = examples.loc[first]
    counts = selected.groupby(["split", "task"]).pair_id.nunique()
    assert counts.loc[("train", "email")] == 200
    assert counts.loc[("train", "qa")] == 200
    assert counts.loc[("validation", "email")] == 100
    assert counts.loc[("validation", "qa")] == 100
    assert selected.groupby("pair_id").size().eq(2).all()


def test_comparison_layers_use_selected_neighborhood_only_when_cached() -> None:
    assert _comparison_layers(26, list(range(48))) == [24, 25, 26, 27, 28]
    assert _comparison_layers(28, [0, 9, 18, 28, 37, 46]) == [28]


def test_density_grid_reports_every_method_and_layer(tmp_path) -> None:
    rows = []
    for layer in (24, 25):
        for method in ("screened_greedy", "gradient_pursuit"):
            for condition, label, score in (
                ("control", 0, -0.1),
                ("attack", 1, 0.1),
            ):
                rows.append(
                    {
                        "split": "validation",
                        "layer": layer,
                        "method": method,
                        "condition": condition,
                        "label": label,
                        "score": score,
                    }
                )
    _plot_density_grid(tmp_path, pd.DataFrame(rows), [24, 25])
    assert (tmp_path / "reconstruction_density_by_layer.png").is_file()


def test_reconstruction_cache_resumes_and_rejects_changed_identity(tmp_path) -> None:
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    activation_path = tmp_path / "activations.dat"
    activations = np.memmap(activation_path, dtype=np.uint16, mode="w+", shape=(2, 1, 2))
    activations[:, 0, :] = tensor_to_bfloat16_bits(hidden)
    activations.flush()
    dictionary = torch.eye(2)
    config = SimpleNamespace(decomposition_batch_size=2, screen_candidates=2)
    selected = {"run_id": "run-a", "manifest_sha256": "manifest-a"}
    kwargs = {
        "output_dir": tmp_path / "output",
        "method": "screened_greedy",
        "layer": 0,
        "sparsity_k": 1,
        "subset_indices": np.array([0, 1]),
        "subset_hash": "subset",
        "layer_position": 0,
        "activations": activations,
        "dictionary": dictionary,
        "config": config,
        "selected": selected,
    }
    first = _load_or_compute_reconstruction(**kwargs)
    second = _load_or_compute_reconstruction(**kwargs)
    torch.testing.assert_close(first["reconstruction"], second["reconstruction"])
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _load_or_compute_reconstruction(**{**kwargs, "selected": {**selected, "run_id": "run-b"}})
