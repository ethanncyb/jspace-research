"""Unit tests for promptguard.jspace_probe (synthetic features, no model)."""

from __future__ import annotations

import pytest
import torch

from promptguard.jspace_probe import JSpaceProbe

LAYERS = [21, 25]
N_CONCEPTS = 32


def make_examples(n: int, *, label: int, seed: int) -> list[dict[int, torch.Tensor]]:
    """Separable toy data: positives spike concept 0 at every layer."""

    generator = torch.Generator().manual_seed(seed + label)
    examples: list[dict[int, torch.Tensor]] = []
    for _ in range(n):
        example = {}
        for layer in LAYERS:
            features = torch.rand(N_CONCEPTS, generator=generator) * 0.1
            if label == 1:
                features[0] = 0.9
            example[layer] = features
        examples.append(example)
    return examples


def make_dataset(n_per_class: int = 16):
    examples = make_examples(n_per_class, label=0, seed=11) + make_examples(
        n_per_class, label=1, seed=22
    )
    labels = [0] * n_per_class + [1] * n_per_class
    return examples, labels


class TestJSpaceProbeLinear:
    def test_fit_learns_separable_data(self) -> None:
        examples, labels = make_dataset()
        probe = JSpaceProbe(LAYERS, N_CONCEPTS)
        losses = probe.fit(
            examples,
            labels,
            epochs=30,
            learning_rate=1e-2,
            batch_size=8,
            seed=3,
        )
        assert losses[-1] < losses[0]
        metrics = probe.evaluate(examples, labels)
        assert metrics.accuracy == pytest.approx(1.0)
        assert metrics.auc == pytest.approx(1.0)
        assert metrics.unsafe_recall == pytest.approx(1.0)
        assert metrics.benign_pass_rate == pytest.approx(1.0)

    def test_score_shape_and_range(self) -> None:
        examples, _ = make_dataset(2)
        probe = JSpaceProbe(LAYERS, N_CONCEPTS)
        aggregate, layer_values = probe.score(examples[0])
        assert 0.0 <= float(aggregate.reshape(-1)[0]) <= 1.0
        assert sorted(layer_values) == LAYERS

    def test_feature_direction_is_unit_norm(self) -> None:
        probe = JSpaceProbe(LAYERS, N_CONCEPTS)
        direction = probe.feature_direction(21)
        assert direction.norm().item() == pytest.approx(1.0)
        assert direction.shape == (N_CONCEPTS,)

    def test_save_load_round_trip(self, tmp_path) -> None:
        examples, _ = make_dataset(2)
        probe = JSpaceProbe(LAYERS, N_CONCEPTS, aggregate="max_logits")
        before = float(probe.score(examples[0])[0].reshape(-1)[0])
        path = tmp_path / "probe.pt"
        probe.save(path, pooling="max")
        loaded = JSpaceProbe.load(path)
        after = float(loaded.score(examples[0])[0].reshape(-1)[0])
        assert before == pytest.approx(after)
        assert loaded.aggregate == "max_logits"
        assert loaded.head == "linear"

    def test_missing_layer_rejected(self) -> None:
        examples, _ = make_dataset(1)
        probe = JSpaceProbe(LAYERS, N_CONCEPTS)
        with pytest.raises(ValueError, match="missing layers"):
            probe.score({21: examples[0][21]})


class TestJSpaceProbeMLP:
    def test_fit_smoke(self) -> None:
        examples, labels = make_dataset(8)
        probe = JSpaceProbe(LAYERS, N_CONCEPTS, head="mlp", mlp_hidden=16)
        losses = probe.fit(examples, labels, epochs=3, batch_size=8, seed=5)
        assert len(losses) == 3
        score = float(probe.score(examples[0])[0].reshape(-1)[0])
        assert 0.0 <= score <= 1.0

    def test_feature_direction_undefined_for_mlp(self) -> None:
        probe = JSpaceProbe(LAYERS, N_CONCEPTS, head="mlp")
        with pytest.raises(ValueError, match="linear"):
            probe.feature_direction(21)

    def test_save_load_round_trip(self, tmp_path) -> None:
        examples, _ = make_dataset(2)
        probe = JSpaceProbe(LAYERS, N_CONCEPTS, head="mlp", mlp_hidden=16)
        before = float(probe.score(examples[0])[0].reshape(-1)[0])
        path = tmp_path / "probe_mlp.pt"
        probe.save(path)
        loaded = JSpaceProbe.load(path)
        assert loaded.head == "mlp"
        assert float(loaded.score(examples[0])[0].reshape(-1)[0]) == pytest.approx(
            before
        )

    def test_unknown_head_rejected(self) -> None:
        with pytest.raises(ValueError, match="head"):
            JSpaceProbe(LAYERS, N_CONCEPTS, head="transformer")
