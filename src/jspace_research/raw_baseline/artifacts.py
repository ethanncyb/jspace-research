from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..phase1.artifacts import Phase1Handoff, load_phase1_handoff
from ..phase1.jspace import read_bfloat16_bits
from ..runtime import read_json
from .config import RawBaselineConfig


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class RawActivationCache:
    handoff: Phase1Handoff
    residuals: np.memmap
    selected_position: int
    layers: tuple[int, ...]
    width: int

    def read(self, indices: np.ndarray) -> np.ndarray:
        bits = np.asarray(self.residuals[indices, self.selected_position, :]).copy()
        return read_bfloat16_bits(bits).numpy().astype(np.float32, copy=False)


def load_raw_activation_cache(config: RawBaselineConfig) -> RawActivationCache:
    handoff = load_phase1_handoff(config.phase1_selected_path, config.phase1)
    root = config.phase1_selected_path.parent
    activation = handoff.metadata["artifacts"]["activations"]
    metadata = read_json(_resolve(root, activation["metadata"]))
    number_examples = int(metadata["number_examples"])
    layers = tuple(int(value) for value in metadata["layers"])
    width = int(metadata["d_model"])
    selected_position = int(activation["layer_position"])
    if number_examples != len(handoff.examples):
        raise RuntimeError("Raw activation cache example count does not match Phase 1")
    if not 0 <= selected_position < len(layers):
        raise RuntimeError("Raw activation cache selected-layer position is invalid")
    if layers[selected_position] != handoff.selected_layer:
        raise RuntimeError("Raw activation cache does not point to the frozen selected layer")
    done = np.load(_resolve(root, activation["completion"]), allow_pickle=False)
    if done.shape != (number_examples,) or done.dtype != np.bool_ or not bool(done.all()):
        raise RuntimeError("Raw activation cache is incomplete")
    path = _resolve(root, activation["residuals"])
    shape = (number_examples, len(layers), width)
    expected_bytes = int(np.prod(shape)) * np.dtype(np.uint16).itemsize
    if path.stat().st_size != expected_bytes:
        raise RuntimeError("Raw activation cache file size does not match metadata")
    residuals = np.memmap(path, dtype=np.uint16, mode="r", shape=shape)
    return RawActivationCache(
        handoff=handoff,
        residuals=residuals,
        selected_position=selected_position,
        layers=layers,
        width=width,
    )


def load_raw_detector(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    value = torch.load(target, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a detector mapping: {target}")
    expected = {
        "schema_version": 1,
        "artifact_type": "raw_residual_detector",
        "frozen": True,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError(f"Unsupported raw detector artifact: {target}")
    if value.get("detector") not in {"mean", "logistic"}:
        raise ValueError(f"Unknown raw detector type: {target}")
    for key in (
        "phase1_run_id",
        "raw_baseline_config_sha256",
        "selected_layer",
        "raw_width",
        "threshold",
    ):
        if key not in value:
            raise ValueError(f"Raw detector artifact is incomplete: {target}")
    width = int(value["raw_width"])
    if width <= 0 or not np.isfinite(float(value["threshold"])):
        raise ValueError(f"Raw detector metadata is invalid: {target}")
    if value["detector"] == "mean":
        tensors = [value.get(key) for key in ("mu_clean", "d_raw", "d_unit")]
        if not all(
            isinstance(tensor, torch.Tensor)
            and tensor.shape == (width,)
            and bool(torch.isfinite(tensor).all())
            for tensor in tensors
        ):
            raise ValueError(f"Raw mean detector vectors are invalid: {target}")
        if not np.isfinite(float(value.get("d_norm", 0))) or float(value["d_norm"]) <= 0:
            raise ValueError(f"Raw mean detector norm is invalid: {target}")
    else:
        weights = value.get("weights")
        if (
            not isinstance(weights, torch.Tensor)
            or weights.shape != (width,)
            or not bool(torch.isfinite(weights).all())
            or not np.isfinite(float(value.get("intercept", np.nan)))
        ):
            raise ValueError(f"Raw logistic detector parameters are invalid: {target}")
        settings = value.get("settings")
        if not isinstance(settings, dict) or settings.get("solver") != "liblinear":
            raise ValueError(f"Raw logistic settings are invalid: {target}")
    return value
