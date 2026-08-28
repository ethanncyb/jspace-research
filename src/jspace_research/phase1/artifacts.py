from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import read_json, sha256_file
from .config import Phase1Config
from .data import expand_examples, read_jsonl, validate_pair_manifest
from .jspace import read_bfloat16_bits


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Artifact path escapes the Phase 1 run directory: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase 1 artifact: {path}")
    return path


def load_selected_layer(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and verify a frozen Phase 1 selection without notebook state."""

    selected_path = Path(path).expanduser().resolve()
    metadata = read_json(selected_path)
    if metadata.get("schema_version") != 1 or metadata.get("phase") != 1:
        raise ValueError(f"Unsupported selected-layer artifact schema: {selected_path}")
    if metadata.get("frozen") is not True:
        raise ValueError(f"Selected layer is not marked frozen: {selected_path}")

    root = selected_path.parent
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("selected_layer.json does not contain an artifact map")

    provenance = read_json(_resolve_artifact(root, artifacts["provenance"]))
    if provenance.get("run_id") != metadata.get("run_id"):
        raise RuntimeError("Selected-layer and provenance run IDs do not match")
    if provenance.get("selected_layer") != metadata.get("selected_layer"):
        raise RuntimeError("Selected-layer and provenance layer values do not match")

    manifest_path = _resolve_artifact(root, artifacts["pair_manifest"])
    if sha256_file(manifest_path) != metadata.get("manifest_sha256"):
        raise RuntimeError("Selected-layer manifest hash does not match")

    direction_metadata = artifacts.get("direction")
    if not isinstance(direction_metadata, dict):
        raise ValueError("selected_layer.json does not identify the direction artifact")
    direction_path = _resolve_artifact(root, direction_metadata["path"])
    if sha256_file(direction_path) != direction_metadata.get("sha256"):
        raise RuntimeError("Selected-layer direction hash does not match")
    direction = torch.load(direction_path, map_location="cpu", weights_only=True)
    required_direction_keys = {"layer", "mu_clean", "d_raw", "d_norm", "d_unit"}
    if not isinstance(direction, dict) or not required_direction_keys.issubset(direction):
        raise ValueError("Selected-layer direction artifact is incomplete")
    if int(direction["layer"]) != int(metadata["selected_layer"]):
        raise RuntimeError("Selected-layer JSON and direction tensor disagree")

    activations = artifacts.get("activations")
    decomposition = artifacts.get("selected_layer_decomposition")
    if not isinstance(activations, dict) or not isinstance(decomposition, dict):
        raise ValueError("Selected-layer cache references are incomplete")
    activation_metadata = read_json(_resolve_artifact(root, activations["metadata"]))
    if activation_metadata.get("config_sha256") != metadata.get(
        "config_sha256"
    ) or activation_metadata.get("manifest_sha256") != metadata.get("manifest_sha256"):
        raise RuntimeError("Selected-layer activation cache identity does not match")
    _resolve_artifact(root, activations["residuals"])
    _resolve_artifact(root, activations["unembedding"])
    activation_done = np.load(
        _resolve_artifact(root, activations["completion"]), allow_pickle=False
    )
    activation_count = int(activation_metadata.get("number_examples", -1))
    if (
        activation_done.dtype != np.bool_
        or activation_done.shape != (activation_count,)
        or not bool(activation_done.all())
    ):
        raise RuntimeError("Selected-layer activation cache is incomplete")

    decomposition_metadata = read_json(_resolve_artifact(root, decomposition["metadata"]))
    if (
        decomposition_metadata.get("config_sha256") != metadata.get("config_sha256")
        or decomposition_metadata.get("manifest_sha256") != metadata.get("manifest_sha256")
        or decomposition_metadata.get("layer") != metadata.get("selected_layer")
    ):
        raise RuntimeError("Selected-layer decomposition cache identity does not match")
    for key in ("reconstruction", "support_ids", "coefficients"):
        _resolve_artifact(root, decomposition[key])
    decomposition_done = np.load(
        _resolve_artifact(root, decomposition["completion"]), allow_pickle=False
    )
    reconstruction_shape = decomposition_metadata.get("reconstruction_shape", [])
    decomposition_count = int(reconstruction_shape[0]) if reconstruction_shape else -1
    if (
        decomposition_done.dtype != np.bool_
        or decomposition_done.shape != (decomposition_count,)
        or not bool(decomposition_done.all())
    ):
        raise RuntimeError("Selected-layer decomposition cache is incomplete")

    return metadata, direction


@dataclass
class Phase1Handoff:
    metadata: dict[str, Any]
    direction: dict[str, Any]
    examples: list[dict[str, Any]]
    reconstruction: np.memmap
    support_ids: np.memmap
    coefficients: np.memmap
    reconstruction_shape: tuple[int, int]
    sparse_shape: tuple[int, int]

    @property
    def selected_layer(self) -> int:
        return int(self.metadata["selected_layer"])

    @property
    def validation_examples(self) -> list[dict[str, Any]]:
        return [example for example in self.examples if example["split"] == "validation"]

    def reconstructed_jspace(self, example_index: int) -> torch.Tensor:
        if not 0 <= example_index < self.reconstruction_shape[0]:
            raise IndexError(f"Example index is outside the reconstruction cache: {example_index}")
        return read_bfloat16_bits(np.asarray(self.reconstruction[example_index]).copy())


def _memmap_file(
    path: Path,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, int],
) -> np.memmap:
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"Phase 1 cache file size does not match metadata: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def load_phase1_handoff(
    selected_path: str | Path, config: Phase1Config
) -> Phase1Handoff:
    """Load the frozen examples and selected-layer caches for downstream phases."""

    path = Path(selected_path).expanduser().resolve()
    metadata, direction = load_selected_layer(path)
    if metadata.get("config_sha256") != config.identity_hash():
        raise RuntimeError("Phase 1 selection does not match the supplied experiment config")

    root = path.parent
    artifacts = metadata["artifacts"]
    rows = read_jsonl(_resolve_artifact(root, artifacts["pair_manifest"]))
    validate_pair_manifest(rows, config)
    examples = expand_examples(rows)
    if not examples or not any(row["split"] == "train" for row in examples):
        raise RuntimeError("The Phase 1 handoff contains no training examples")
    if not any(row["split"] == "validation" for row in examples):
        raise RuntimeError("The Phase 1 handoff contains no validation examples")

    decomposition = artifacts["selected_layer_decomposition"]
    decomposition_metadata = read_json(
        _resolve_artifact(root, decomposition["metadata"])
    )
    raw_reconstruction_shape = decomposition_metadata.get("reconstruction_shape")
    raw_sparse_shape = decomposition_metadata.get("sparse_shape")
    if not isinstance(raw_reconstruction_shape, list) or len(raw_reconstruction_shape) != 2:
        raise ValueError("Selected-layer reconstruction metadata has an invalid shape")
    if not isinstance(raw_sparse_shape, list) or len(raw_sparse_shape) != 2:
        raise ValueError("Selected-layer sparse metadata has an invalid shape")
    reconstruction_shape = tuple(int(value) for value in raw_reconstruction_shape)
    sparse_shape = tuple(int(value) for value in raw_sparse_shape)
    if reconstruction_shape[0] != len(examples) or sparse_shape[0] != len(examples):
        raise RuntimeError("Manifest and selected-layer cache counts do not match")
    if sparse_shape[1] != int(metadata["decomposition"]["sparsity_k"]):
        raise RuntimeError("Selected-layer sparse cache does not match the frozen sparsity")

    reconstruction = _memmap_file(
        _resolve_artifact(root, decomposition["reconstruction"]),
        dtype=np.dtype(np.uint16),
        shape=reconstruction_shape,
    )
    support_ids = _memmap_file(
        _resolve_artifact(root, decomposition["support_ids"]),
        dtype=np.dtype(np.int32),
        shape=sparse_shape,
    )
    coefficients = _memmap_file(
        _resolve_artifact(root, decomposition["coefficients"]),
        dtype=np.dtype(np.float32),
        shape=sparse_shape,
    )
    return Phase1Handoff(
        metadata=metadata,
        direction=direction,
        examples=examples,
        reconstruction=reconstruction,
        support_ids=support_ids,
        coefficients=coefficients,
        reconstruction_shape=reconstruction_shape,
        sparse_shape=sparse_shape,
    )
