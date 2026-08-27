from __future__ import annotations

import pytest
import torch

from jspace_research.phase1.adapters import validate_lens_for_layers, validate_model_lens


class FakeModel:
    hidden_width = 3
    number_layers = 4


class FakeLens:
    hidden_width = 3
    source_layers = (1, 2)

    def jacobian(self, layer: int) -> torch.Tensor:
        assert layer in self.source_layers
        return torch.eye(3)


def test_model_lens_compatibility_is_explicitly_validated() -> None:
    validate_model_lens(FakeModel(), FakeLens())  # type: ignore[arg-type]

    incompatible = FakeLens()
    incompatible.hidden_width = 4
    with pytest.raises(RuntimeError, match="width mismatch"):
        validate_model_lens(FakeModel(), incompatible)  # type: ignore[arg-type]


def test_cached_layers_must_be_fitted_by_lens() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        validate_lens_for_layers(FakeLens(), 3, [0])  # type: ignore[arg-type]
