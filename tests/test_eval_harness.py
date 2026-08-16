import csv
from dataclasses import asdict

import torch

from eval_harness import evaluate
from probe import DatasetRow, LayerProbeDetector


def test_evaluate_writes_requested_artifacts(tmp_path) -> None:
    rows = [
        DatasetRow("benign", "clean", " ok", 0),
        DatasetRow("attack", "clean", " injected", 1),
    ]
    features = {0: torch.tensor([[-1.0, 0.0], [1.0, 0.0]])}
    artifact = tmp_path / "features.pt"
    torch.save(
        {
            "model_id": "tiny",
            "layers": (0,),
            "hidden_dim": 2,
            "rows": [asdict(row) for row in rows],
            "features": features,
        },
        artifact,
    )
    detector = LayerProbeDetector([0], 2, model_id="tiny")
    with torch.no_grad():
        detector.heads["layer_0"].weight.copy_(torch.tensor([[5.0, 0.0]]))
        detector.heads["layer_0"].bias.zero_()
    checkpoint = tmp_path / "probe.pt"
    detector.save(checkpoint)
    output = tmp_path / "evaluation"
    summary = evaluate({}, artifact, checkpoint, output)
    assert summary["accuracy"] == 1
    for name in (
        "per_layer.csv",
        "per_example.csv",
        "summary.csv",
        "auto_buffer.jsonl",
        "manual_review.csv",
    ):
        assert (output / name).exists()
    with (output / "per_example.csv").open(newline="") as stream:
        evaluated = list(csv.DictReader(stream))
    assert [row["confusion_category"] for row in evaluated] == [
        "true_negative",
        "true_positive",
    ]
