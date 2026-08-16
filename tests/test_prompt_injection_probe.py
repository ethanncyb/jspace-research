import csv
import json

import pytest
import torch

from probe import (
    DatasetRow,
    LayerProbeDetector,
    classification_metrics,
    load_dataset,
    stratified_group_split,
    train_probes,
)
from tests.tiny import TinyDecoder


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv"])
def test_dataset_formats_and_field_mapping(tmp_path, extension: str) -> None:
    raw = [{"key": "a", "prefix": "hello", "suffix": " world", "kind": "benign"}]
    path = tmp_path / f"data.{extension}"
    if extension == "json":
        path.write_text(json.dumps(raw))
    elif extension == "jsonl":
        path.write_text(json.dumps(raw[0]) + "\n")
    else:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=raw[0])
            writer.writeheader()
            writer.writerows(raw)
    rows = load_dataset(
        path,
        field_map={
            "id": "key",
            "clean_prompt": "prefix",
            "appended_text": "suffix",
            "label": "kind",
        },
    )
    assert rows == [DatasetRow("a", "hello", " world", 0, {})]


def test_candidate_prompt_exact_prefix_validation(tmp_path) -> None:
    tokenizer = TinyDecoder().tokenizer
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            [
                {
                    "id": "a",
                    "clean_prompt": "abc",
                    "candidate_prompt": "abcXYZ",
                    "label": 1,
                }
            ]
        )
    )
    row = load_dataset(valid, tokenizer=tokenizer)[0]
    assert row.appended_text
    assert row.metadata["_candidate_suffix_token_ids"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            [{"id": "b", "clean_prompt": "abc", "candidate_prompt": "abd", "label": 1}]
        )
    )
    with pytest.raises(ValueError, match="exact prefix"):
        load_dataset(invalid, tokenizer=tokenizer)


def _rows() -> list[DatasetRow]:
    return [
        DatasetRow(f"n{i}", "p", "a", 0, {"pair_id": f"n{i // 2}"}) for i in range(8)
    ] + [DatasetRow(f"p{i}", "p", "b", 1, {"pair_id": f"p{i // 2}"}) for i in range(8)]


def test_grouped_split_no_leakage_and_seeded() -> None:
    rows = _rows()
    first = stratified_group_split(rows, 0.25, seed=4)
    second = stratified_group_split(rows, 0.25, seed=4)
    assert first == second
    train, validation = first
    assert {rows[i].pair_id for i in train}.isdisjoint(
        {rows[i].pair_id for i in validation}
    )


def test_training_round_trip_compatibility_and_determinism(tmp_path) -> None:
    rows = _rows()
    labels = torch.tensor([row.label for row in rows])
    base = labels.float().mul(4).sub(2).unsqueeze(1)
    features = {
        1: torch.cat((base, base * 0.5, torch.ones_like(base)), dim=1),
        3: torch.cat((base * 0.75, base, -torch.ones_like(base)), dim=1),
    }
    first = train_probes(features, labels, rows, epochs=30, seed=9, model_id="tiny")
    second = train_probes(features, labels, rows, epochs=30, seed=9, model_id="tiny")
    for key, value in first.detector.state_dict().items():
        assert torch.equal(value, second.detector.state_dict()[key])
    path = tmp_path / "probe.pt"
    first.detector.save(path)
    restored = LayerProbeDetector.load(
        path, expected_layers=[1, 3], expected_hidden_dim=3, expected_model_id="tiny"
    )
    assert restored.layers == (1, 3)
    assert torch.equal(
        restored.raw_attack_direction(1), first.detector.raw_attack_direction(1)
    )
    with pytest.raises(ValueError, match="hidden size"):
        LayerProbeDetector.load(path, expected_hidden_dim=4)
    with pytest.raises(ValueError, match="layers"):
        LayerProbeDetector.load(path, expected_layers=[1])


def test_metrics_aggregation_and_distances() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert metrics["accuracy"] == 1
    assert metrics["auroc"] == 1
    assert metrics["false_positive_rate"] == 0
    assert metrics["false_negative_rate"] == 0
    detector = LayerProbeDetector([1, 2], 2, aggregation="mean")
    assert detector.aggregate({1: 0.2, 2: 0.8}) == pytest.approx(0.5)
    detector.set_statistics(
        1,
        mean=torch.zeros(2),
        std=torch.ones(2),
        benign_reference=torch.tensor([1.0, 0.0]),
    )
    assert detector.benign_distance(
        1, torch.tensor([1.0, 0.0])
    ).item() == pytest.approx(0)
