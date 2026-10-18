from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from jspace_research.phase1.jspace import (
    build_normalized_dictionary,
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


def test_streamed_screening_matches_full_dictionary() -> None:
    generator = torch.Generator().manual_seed(42)
    dictionary = torch.nn.functional.normalize(
        torch.randn(23, 7, generator=generator), dim=1
    )
    hidden = torch.randn(2, 7, generator=generator)
    full = screened_nonnegative_pursuit(
        hidden, dictionary, sparsity_k=3, screen_candidates=6
    )
    streamed = screened_nonnegative_pursuit(
        hidden,
        dictionary,
        sparsity_k=3,
        screen_candidates=6,
        scoring_device=torch.device("cpu"),
        screen_chunk_size=5,
    )
    torch.testing.assert_close(streamed[0], full[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(streamed[1], full[1])
    torch.testing.assert_close(streamed[2], full[2], atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dictionary_can_be_built_directly_on_cpu() -> None:
    generator = torch.Generator().manual_seed(42)
    jacobian = torch.randn(7, 7, generator=generator)
    unembedding = torch.randn(23, 7, generator=generator)
    kwargs = {
        "jacobian": jacobian,
        "unembedding": unembedding,
        "layer": 0,
        "device": torch.device("cuda:0"),
        "chunk_size": 5,
    }
    gpu_dictionary = build_normalized_dictionary(**kwargs)
    cpu_dictionary = build_normalized_dictionary(
        **kwargs, storage_device=torch.device("cpu")
    )
    assert cpu_dictionary.device.type == "cpu"
    torch.testing.assert_close(cpu_dictionary, gpu_dictionary.cpu(), atol=0, rtol=0)
    hidden = torch.randn(1, 7, generator=generator)
    full = screened_nonnegative_pursuit(
        hidden, gpu_dictionary, sparsity_k=3, screen_candidates=6
    )
    streamed = screened_nonnegative_pursuit(
        hidden,
        cpu_dictionary,
        sparsity_k=3,
        screen_candidates=6,
        scoring_device=torch.device("cuda:0"),
        screen_chunk_size=5,
    )
    torch.testing.assert_close(streamed[0], full[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(streamed[1], full[1])
    torch.testing.assert_close(streamed[2], full[2], atol=1e-4, rtol=1e-4)
    repeated_gpu = torch.cat((gpu_dictionary, gpu_dictionary[:3]), dim=0)
    repeated_cpu = repeated_gpu.cpu()
    tied_hidden = repeated_cpu[0:1].float()
    tied_full = screened_nonnegative_pursuit(
        tied_hidden, repeated_gpu, sparsity_k=3, screen_candidates=6
    )
    tied_streamed = screened_nonnegative_pursuit(
        tied_hidden,
        repeated_cpu,
        sparsity_k=3,
        screen_candidates=6,
        scoring_device=torch.device("cuda:0"),
        screen_chunk_size=5,
    )
    torch.testing.assert_close(tied_streamed[1], tied_full[1])


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
