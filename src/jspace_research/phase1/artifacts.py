from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import read_json, sha256_file


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
