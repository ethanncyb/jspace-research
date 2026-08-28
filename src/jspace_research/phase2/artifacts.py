from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..phase1.artifacts import load_selected_layer
from ..phase1.cache import read_json
from ..phase1.data import expand_examples, read_jsonl, validate_pair_manifest
from ..phase1.jspace import read_bfloat16_bits
from .config import Phase2Config


def _resolve(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Artifact path escapes the Phase 1 run directory: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase 1 artifact: {path}")
    return path


@dataclass
class Phase1Handoff:
    metadata: dict[str, Any]
    validation_examples: list[dict[str, Any]]
    reconstruction: np.memmap
    reconstruction_shape: tuple[int, int]

    @property
    def selected_layer(self) -> int:
        return int(self.metadata["selected_layer"])

    def reconstructed_jspace(self, example_index: int) -> torch.Tensor:
        if not 0 <= example_index < self.reconstruction_shape[0]:
            raise IndexError(f"Example index is outside the reconstruction cache: {example_index}")
        return read_bfloat16_bits(np.asarray(self.reconstruction[example_index]).copy())


def load_phase1_handoff(config: Phase2Config) -> Phase1Handoff:
    metadata, _ = load_selected_layer(config.phase1_selected_path)
    if metadata.get("config_sha256") != config.phase1.identity_hash():
        raise RuntimeError("Phase 1 selection does not match the supplied experiment config")

    root = config.phase1_selected_path.parent
    artifacts = metadata["artifacts"]
    rows = read_jsonl(_resolve(root, artifacts["pair_manifest"]))
    validate_pair_manifest(rows, config.phase1)
    examples = expand_examples(rows)
    validation_examples = [row for row in examples if row["split"] == "validation"]
    if not validation_examples:
        raise RuntimeError("The Phase 1 handoff contains no validation examples")

    decomposition = artifacts["selected_layer_decomposition"]
    decomposition_metadata = read_json(_resolve(root, decomposition["metadata"]))
    raw_shape = decomposition_metadata.get("reconstruction_shape")
    if not isinstance(raw_shape, list) or len(raw_shape) != 2:
        raise ValueError("Selected-layer reconstruction metadata has an invalid shape")
    shape = (int(raw_shape[0]), int(raw_shape[1]))
    if shape[0] != len(examples):
        raise RuntimeError("Manifest and selected-layer reconstruction counts do not match")
    reconstruction_path = _resolve(root, decomposition["reconstruction"])
    expected_bytes = shape[0] * shape[1] * np.dtype(np.uint16).itemsize
    if reconstruction_path.stat().st_size != expected_bytes:
        raise RuntimeError("Selected-layer reconstruction file size does not match metadata")
    reconstruction = np.memmap(
        reconstruction_path,
        dtype=np.uint16,
        mode="r",
        shape=shape,
    )
    return Phase1Handoff(metadata, validation_examples, reconstruction, shape)
