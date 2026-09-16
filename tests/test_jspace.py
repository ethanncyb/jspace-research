from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from jspace_research.phase1.jspace import (
    compute_layer_metrics,
    direction_scores,
    learn_task_balanced_direction,
    nonnegative_gradient_pursuit,
    screened_nonnegative_pursuit,
    select_layer,
    select_task_macro_balanced_threshold,
    threshold_metrics,
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


def test_sparse_pursuit_refits_nonorthogonal_positive_atoms() -> None:
    diagonal = 2**-0.5
    dictionary = torch.tensor([[1.0, 0.0], [diagonal, diagonal]])
    hidden = dictionary[0:1] + 2 * dictionary[1:2]
    reconstruction, token_ids, coefficients = screened_nonnegative_pursuit(
        hidden, dictionary, sparsity_k=2, screen_candidates=2
    )
    torch.testing.assert_close(reconstruction, hidden, atol=1e-5, rtol=1e-5)
    assert set(token_ids[0].tolist()) == {0, 1}
    assert bool((coefficients >= 0).all())


def test_sparse_pursuit_excludes_nonpositive_screen_candidates() -> None:
    dictionary = torch.tensor([[0.8, 0.6], [0.0, -1.0]])
    hidden = torch.tensor([[1.0, 0.0]])
    _, token_ids, _ = screened_nonnegative_pursuit(
        hidden, dictionary, sparsity_k=2, screen_candidates=2
    )
    assert token_ids[0].tolist() == [0, -1]


def test_gradient_pursuit_is_nonnegative_and_respects_k() -> None:
    dictionary = torch.eye(8)
    hidden = torch.tensor([[0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0]])
    reconstruction, token_ids, coefficients = nonnegative_gradient_pursuit(
        hidden, dictionary, sparsity_k=3
    )
    torch.testing.assert_close(reconstruction, hidden, atol=1e-5, rtol=1e-5)
    assert int((token_ids[0] >= 0).sum()) <= 3
    assert bool((coefficients >= 0).all())


def test_gradient_pursuit_improves_over_one_atom() -> None:
    dictionary = torch.tensor([[1.0, 0.0], [2**-0.5, 2**-0.5], [0.0, 1.0]])
    hidden = torch.tensor([[1.0, 1.0]])
    one, _, _ = nonnegative_gradient_pursuit(hidden, dictionary, sparsity_k=1)
    two, _, _ = nonnegative_gradient_pursuit(hidden, dictionary, sparsity_k=2)
    assert torch.linalg.vector_norm(hidden - two) <= torch.linalg.vector_norm(hidden - one)


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
    assert set(metrics.scope) == {"task", "macro"}
    selected = select_layer(metrics)
    assert selected.layer == 3
    assert selected.auprc == pytest.approx(1.0)


def test_training_threshold_uses_task_macro_balanced_accuracy() -> None:
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    scores = np.array([0.0, 0.1, 0.8, 0.9, 0.2, 0.3, 0.7, 1.0])
    tasks = ["a"] * 4 + ["b"] * 4
    threshold, value = select_task_macro_balanced_threshold(labels, scores, tasks)
    assert threshold == pytest.approx(0.7)
    assert value == pytest.approx(1.0)
    metrics = threshold_metrics(labels, scores, threshold)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
