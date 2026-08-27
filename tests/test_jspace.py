from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from jspace_research.phase1.jspace import (
    compute_layer_metrics,
    direction_scores,
    learn_task_balanced_direction,
    screened_nonnegative_pursuit,
    select_layer,
)


def test_sparse_pursuit_recovers_positive_identity_atoms() -> None:
    dictionary = torch.eye(8)
    hidden = torch.tensor([[0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0]])
    reconstruction, token_ids, coefficients = screened_nonnegative_pursuit(
        hidden, dictionary, sparsity_k=3, screen_candidates=8
    )
    torch.testing.assert_close(reconstruction, hidden, atol=1e-5, rtol=1e-5)
    assert int((token_ids[0] >= 0).sum()) <= 3
    assert bool((coefficients >= 0).all())


def test_task_balanced_direction_and_scores() -> None:
    representations = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 0.0],
            [100.0, 2.0],
            [100.0, 0.0],
        ]
    )
    labels = np.array([1, 0, 1, 0, 1, 0])
    tasks = ["a", "a", "a", "a", "b", "b"]
    artifact = learn_task_balanced_direction(representations, labels, tasks)
    expected_raw = torch.tensor([1.5, 1.0])
    torch.testing.assert_close(artifact["d_raw"], expected_raw)
    scores = direction_scores(representations, artifact["mu_clean"], artifact["d_unit"])
    assert scores[0] > scores[1]
    assert scores[4] > scores[5]


def test_metrics_and_layer_tie_breaking() -> None:
    rows = []
    for layer in (3, 7):
        for task in ("a", "b"):
            rows.extend(
                [
                    {"layer": layer, "task": task, "label": 0, "score": 0.0},
                    {"layer": layer, "task": task, "label": 1, "score": 1.0},
                ]
            )
    metrics = compute_layer_metrics(pd.DataFrame(rows), [3, 7], ["a", "b"], {"a": "A", "b": "B"})
    selected = select_layer(metrics)
    assert selected.layer == 3
    assert selected.auprc == pytest.approx(1.0)
