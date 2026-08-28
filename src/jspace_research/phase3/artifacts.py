from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_detector(path: str | Path) -> dict[str, Any]:
    """Load one frozen Phase 3 detector and verify its direct contents."""

    detector_path = Path(path).expanduser().resolve()
    value = torch.load(detector_path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a detector mapping: {detector_path}")
    if value.get("schema_version") != 1 or value.get("phase") != 3:
        raise ValueError(f"Unsupported detector artifact schema: {detector_path}")
    if value.get("frozen") is not True or value.get("detector") not in {
        "mean",
        "logistic",
    }:
        raise ValueError(f"Detector artifact is not frozen or recognized: {detector_path}")
    common = {
        "phase1_run_id",
        "phase3_config_sha256",
        "selected_layer",
        "decomposition",
        "threshold",
    }
    if not common.issubset(value):
        raise ValueError(f"Detector artifact is incomplete: {detector_path}")
    if not isinstance(value["phase1_run_id"], str) or not value["phase1_run_id"]:
        raise ValueError(f"Detector Phase 1 identity is invalid: {detector_path}")
    config_hash = value["phase3_config_sha256"]
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise ValueError(f"Detector Phase 3 identity is invalid: {detector_path}")
    if not isinstance(value["selected_layer"], int) or value["selected_layer"] < 0:
        raise ValueError(f"Detector selected layer is invalid: {detector_path}")
    decomposition = value["decomposition"]
    if not isinstance(decomposition, dict) or decomposition.get("sparsity_k") != 25:
        raise ValueError(f"Detector decomposition is invalid: {detector_path}")
    threshold = float(value["threshold"])
    if not torch.isfinite(torch.tensor(threshold)):
        raise ValueError(f"Detector threshold must be finite: {detector_path}")

    if value["detector"] == "mean":
        required = {"mu_clean", "d_raw", "d_unit", "d_norm"}
        if not required.issubset(value):
            raise ValueError(f"Mean detector artifact is incomplete: {detector_path}")
        vectors = [value[key] for key in ("mu_clean", "d_raw", "d_unit")]
        if not all(isinstance(vector, torch.Tensor) and vector.ndim == 1 for vector in vectors):
            raise ValueError(f"Mean detector vectors are invalid: {detector_path}")
        if len({int(vector.numel()) for vector in vectors}) != 1:
            raise ValueError(f"Mean detector vector shapes disagree: {detector_path}")
        if not all(bool(torch.isfinite(vector).all()) for vector in vectors):
            raise ValueError(f"Mean detector vectors must be finite: {detector_path}")
        if float(value["d_norm"]) <= 0:
            raise ValueError(f"Mean detector norm must be positive: {detector_path}")
    else:
        required = {"feature_token_ids", "weights", "intercept", "settings"}
        if not required.issubset(value):
            raise ValueError(f"Logistic detector artifact is incomplete: {detector_path}")
        feature_ids = value["feature_token_ids"]
        weights = value["weights"]
        if not isinstance(feature_ids, torch.Tensor) or not isinstance(weights, torch.Tensor):
            raise ValueError(f"Logistic detector tensors are invalid: {detector_path}")
        if feature_ids.ndim != 1 or weights.ndim != 1 or feature_ids.shape != weights.shape:
            raise ValueError(f"Logistic detector feature shapes disagree: {detector_path}")
        if (
            feature_ids.numel() == 0
            or bool((feature_ids < 0).any())
            or not bool((feature_ids[1:] > feature_ids[:-1]).all())
        ):
            raise ValueError(f"Logistic feature token IDs must be sorted and unique: {detector_path}")
        if not bool(torch.isfinite(weights).all()) or not torch.isfinite(
            torch.tensor(float(value["intercept"]))
        ):
            raise ValueError(f"Logistic detector parameters must be finite: {detector_path}")
        settings = value["settings"]
        if not isinstance(settings, dict) or settings.get("solver") != "liblinear":
            raise ValueError(f"Logistic detector settings are invalid: {detector_path}")
    return value
