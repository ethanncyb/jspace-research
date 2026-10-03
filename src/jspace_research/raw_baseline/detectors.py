from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import load_raw_detector


@dataclass(frozen=True)
class RawDetectors:
    mean: dict[str, Any]
    logistic: dict[str, Any]
    dictionary: None = None

    @classmethod
    def load(cls, directory: str | Path, phase1_metadata: dict[str, Any]) -> RawDetectors:
        root = Path(directory).expanduser().resolve()
        mean = load_raw_detector(root / "raw_mean_detector.pt")
        logistic = load_raw_detector(root / "raw_logistic_detector.pt")
        for key, expected in (
            ("phase1_run_id", phase1_metadata["run_id"]),
            ("selected_layer", phase1_metadata["selected_layer"]),
        ):
            if mean[key] != expected or logistic[key] != expected:
                raise RuntimeError(f"Raw detector {key} does not match Phase 1")
        if (
            mean["raw_baseline_config_sha256"] != logistic["raw_baseline_config_sha256"]
            or mean["raw_width"] != logistic["raw_width"]
        ):
            raise RuntimeError("Raw detector artifacts do not match each other")
        return cls(mean=mean, logistic=logistic)

    def score(self, residual: torch.Tensor, dictionary: Any | None = None) -> dict[str, Any]:
        vector = residual.detach().to("cpu", dtype=torch.float32).reshape(-1)
        width = int(self.mean["raw_width"])
        if vector.shape != (width,) or not bool(torch.isfinite(vector).all()):
            raise RuntimeError("Selected-layer raw residual is invalid")
        mean_score = float(
            torch.dot(
                self.mean["d_unit"].float(),
                vector - self.mean["mu_clean"].float(),
            )
        )
        logistic_score = float(self.logistic["intercept"]) + float(
            np.dot(
                self.logistic["weights"].numpy().astype(np.float64),
                vector.numpy().astype(np.float64),
            )
        )
        return {
            "mean_score": mean_score,
            "mean_prediction": mean_score >= float(self.mean["threshold"]),
            "logistic_score": logistic_score,
            "logistic_prediction": logistic_score >= float(self.logistic["threshold"]),
        }
