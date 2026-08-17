from __future__ import annotations

import pytest
import torch

from promptguard.benchmark_setup import BenchmarkExample
from promptguard.drift_probe import DriftProbe
from promptguard.layer_diagnostic import build_diagnostic_prompts, diagnose_layers


def test_diagnostic_reports_each_family_and_layer_with_raw_norms():
    payloads = [BenchmarkExample("p1", "test", "unsafe", "harmful request")]
    prompts = build_diagnostic_prompts(payloads)
    assert {prompt.attack_family for prompt in prompts} == {
        "raw_control",
        "translated",
        "acrostic_encoding",
    }
    probe = DriftProbe([1, 3], 2)
    with torch.no_grad():
        for classifier in probe.classifiers.values():
            classifier.weight[:] = torch.tensor([[1.0, 0.0]])
            classifier.bias.zero_()

    def extractor(_baseline, prompt):
        value = 2.0 if prompt == "harmful request" else -2.0
        return {
            1: torch.tensor([[value, 0.0]]),
            3: torch.tensor([[value / 2, 0.0]]),
        }

    rows, summaries = diagnose_layers(
        object(),
        probe,
        prompts,
        baseline="baseline",
        pooling="last_token",
        threshold=0.5,
        extractor=extractor,
    )
    assert len(rows) == 6
    assert len(summaries) == 6
    raw = next(
        row for row in rows if row.attack_family == "raw_control" and row.layer == 1
    )
    encoded = next(
        row for row in rows if row.attack_family == "translated" and row.layer == 1
    )
    assert raw.drift_score > 0.5
    assert encoded.drift_score < 0.5
    assert raw.delta_l2_norm == pytest.approx(2.0)
