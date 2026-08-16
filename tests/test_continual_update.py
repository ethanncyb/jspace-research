import csv

import pytest
import torch

from continual_update import (
    ContinualUpdateStore,
    FeatureExample,
    FlaggedDeltaRecord,
    cap_pseudo_labeled,
    ewc_penalty,
    fine_tune_and_publish,
)
from probe import LayerProbeDetector


def _record(sample: str, score: float) -> FlaggedDeltaRecord:
    return FlaggedDeltaRecord(sample, "prompt", score, 0, "v1", torch.ones(2))


def test_strict_routing_deduplication_and_review_log(tmp_path) -> None:
    store = ContinualUpdateStore(
        tmp_path / "auto.jsonl", tmp_path / "review.csv", strict_confidence=0.995
    )
    assert store.route(_record("auto", 0.995)) == "auto_positive"
    assert store.route(_record("review", 0.8)) == "manual_review"
    assert store.route(_record("review", 0.8)) == "duplicate"
    assert len(store.auto_records()) == 1
    with (tmp_path / "review.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert "label" not in rows[0]  # review routing never fabricates benign labels


def _example(name: str, label: int, value: float) -> FeatureExample:
    return FeatureExample(name, label, {0: torch.tensor([value, 1.0])})


def test_pseudo_label_batch_cap() -> None:
    trusted = [_example(str(i), i % 2, float(i)) for i in range(8)]
    pseudo = [_example(f"p{i}", 1, 2.0) for i in range(20)]
    selected = cap_pseudo_labeled(trusted, pseudo, max_fraction=0.25, seed=1)
    assert len(selected) == 2
    assert len(selected) / (len(trusted) + len(selected)) <= 0.25


def test_ewc_penalty_zero_at_anchor_and_positive_after_change() -> None:
    detector = LayerProbeDetector([0], 2)
    anchor = {
        name: value.detach().clone() for name, value in detector.named_parameters()
    }
    fisher = {
        name: torch.ones_like(value) for name, value in detector.named_parameters()
    }
    assert ewc_penalty(detector, fisher, anchor).item() == 0
    with torch.no_grad():
        detector.heads["layer_0"].weight.add_(1)
    assert ewc_penalty(detector, fisher, anchor).item() > 0


def test_rollback_on_failed_safeguard_and_atomic_publish(tmp_path) -> None:
    detector = LayerProbeDetector([0], 2)
    original = {key: value.clone() for key, value in detector.state_dict().items()}
    trusted = [
        _example("n0", 0, -1),
        _example("n1", 0, -2),
        _example("p0", 1, 1),
        _example("p1", 1, 2),
    ]
    pseudo = [_example("pseudo", 1, 3)]
    destination = tmp_path / "probe.pt"
    failed = fine_tune_and_publish(
        detector,
        trusted,
        pseudo,
        destination,
        safeguard=lambda _detector: (False, {"reason": "regression"}),
        epochs=2,
    )
    assert not failed.published
    assert failed.rollback_path.exists()
    assert not destination.exists()
    for key, value in original.items():
        assert torch.equal(value, detector.state_dict()[key])

    passed = fine_tune_and_publish(
        detector,
        trusted,
        pseudo,
        destination,
        safeguard=lambda _detector: (True, {"accuracy": 1.0}),
        epochs=2,
    )
    assert passed.published
    assert destination.exists()
    assert LayerProbeDetector.load(destination).hidden_dim == 2


def test_automatic_updates_reject_negative_pseudo_labels(tmp_path) -> None:
    detector = LayerProbeDetector([0], 2)
    trusted = [_example("n", 0, -1), _example("p", 1, 1)]
    with pytest.raises(ValueError, match="positive"):
        fine_tune_and_publish(
            detector,
            trusted,
            [_example("bad", 0, -2)],
            tmp_path / "probe.pt",
            safeguard=lambda _: (True, {}),
            epochs=1,
        )
