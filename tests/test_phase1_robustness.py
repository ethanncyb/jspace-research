from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from jspace_research.phase1.jspace import tensor_to_bfloat16_bits
from jspace_research.phase1.robustness import (
    _load_or_compute_pilot,
    _pilot_indices,
    _pilot_layers,
    _select_ba_layer,
)


def test_pilot_subset_is_deterministic_and_pair_complete() -> None:
    rows = []
    example_index = 0
    for split, pair_count in (("train", 60), ("validation", 55)):
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
    first = _pilot_indices(examples, ("email", "qa"))
    second = _pilot_indices(examples, ("email", "qa"))
    np.testing.assert_array_equal(first, second)
    selected = examples.loc[first]
    counts = selected.groupby(["split", "task"]).pair_id.nunique()
    assert set(counts.tolist()) == {50}
    assert selected.groupby("pair_id").size().eq(2).all()


def test_pilot_layers_use_selected_neighborhood_only_when_cached() -> None:
    assert _pilot_layers(26, list(range(48))) == [24, 25, 26, 27, 28]
    assert _pilot_layers(28, [0, 9, 18, 28, 37, 46]) == [28]


def test_balanced_accuracy_layer_ties_choose_lower_layer() -> None:
    metrics = pd.DataFrame(
        [
            {"layer": 7, "scope": "macro", "balanced_accuracy": 0.8},
            {"layer": 3, "scope": "macro", "balanced_accuracy": 0.8},
        ]
    )
    assert _select_ba_layer(metrics) == 3


def test_pilot_condition_cache_resumes_and_rejects_changed_identity(tmp_path) -> None:
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
    first = _load_or_compute_pilot(**kwargs)
    second = _load_or_compute_pilot(**kwargs)
    torch.testing.assert_close(first["reconstruction"], second["reconstruction"])
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _load_or_compute_pilot(**{**kwargs, "selected": {**selected, "run_id": "run-b"}})
