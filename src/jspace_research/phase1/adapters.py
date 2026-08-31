from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ..model import HuggingFaceModelAdapter
from ..model import load_tokenizer as load_tokenizer
from .cache import sha256_file
from .config import Phase1Config


class JacobianLensAdapter:
    """Small boundary around the pinned Jacobian-lens artifact."""

    def __init__(self, lens: Any, path: Path) -> None:
        self._lens = lens
        self.path = path

    @classmethod
    def load(cls, config: Phase1Config) -> JacobianLensAdapter:
        from huggingface_hub import hf_hub_download
        from jlens import JacobianLens

        path = Path(
            hf_hub_download(
                repo_id=config.lens.repository,
                filename=config.lens.filename,
                revision=config.lens.revision,
            )
        )
        actual = sha256_file(path)
        if actual != config.lens.sha256:
            raise RuntimeError(
                f"Lens SHA-256 mismatch: expected {config.lens.sha256}, found {actual}"
            )
        return cls(JacobianLens.load(str(path)), path)

    @property
    def hidden_width(self) -> int:
        return int(self._lens.d_model)

    @property
    def source_layers(self) -> tuple[int, ...]:
        return tuple(int(layer) for layer in self._lens.source_layers)

    def jacobian(self, layer: int) -> torch.Tensor:
        if layer not in self.source_layers:
            raise ValueError(f"Layer {layer} is not fitted by the configured J-lens")
        return self._lens.jacobians[layer]


def validate_model_lens(
    model: HuggingFaceModelAdapter,
    lens: JacobianLensAdapter,
) -> None:
    if model.hidden_width != lens.hidden_width:
        raise RuntimeError(
            f"Model/lens width mismatch: {model.hidden_width} != {lens.hidden_width}"
        )
    if not lens.source_layers:
        raise RuntimeError("The configured J-lens contains no fitted source layers")
    if min(lens.source_layers) < 0 or max(lens.source_layers) >= model.number_layers:
        raise RuntimeError("The fitted J-lens contains an out-of-range source layer")
    validate_lens_for_layers(lens, model.hidden_width, lens.source_layers)


def validate_lens_for_layers(
    lens: JacobianLensAdapter,
    hidden_width: int,
    layers: Sequence[int],
) -> None:
    if lens.hidden_width != hidden_width:
        raise RuntimeError(
            f"Cached model/lens width mismatch: {hidden_width} != {lens.hidden_width}"
        )
    if not set(layers).issubset(lens.source_layers):
        raise RuntimeError("A cached layer is not fitted by the configured J-lens")
    for layer in layers:
        jacobian = lens.jacobian(layer)
        if tuple(jacobian.shape) != (hidden_width, hidden_width):
            raise RuntimeError(
                f"J-lens layer {layer} has incompatible shape {tuple(jacobian.shape)}"
            )
