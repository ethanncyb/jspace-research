from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..phase1.jspace import screened_nonnegative_pursuit
from ..phase3.artifacts import load_detector


@dataclass(frozen=True)
class FrozenDetectors:
    mean: dict[str, Any]
    logistic: dict[str, Any]
    dictionary: torch.Tensor | None = None

    @classmethod
    def load(cls, phase3_dir: str | Path, phase1_metadata: dict[str, Any]) -> FrozenDetectors:
        root = Path(phase3_dir).expanduser().resolve()
        mean = load_detector(root / "mean_detector.pt")
        logistic = load_detector(root / "logistic_detector.pt")
        for key in ("phase1_run_id", "selected_layer", "decomposition"):
            expected = (
                phase1_metadata["run_id"] if key == "phase1_run_id" else phase1_metadata[key]
            )
            if mean[key] != expected or logistic[key] != expected:
                raise RuntimeError(f"Phase 3 detector {key} does not match Phase 1")
        if mean["phase3_config_sha256"] != logistic["phase3_config_sha256"]:
            raise RuntimeError("Phase 3 detector identities do not match")
        return cls(mean=mean, logistic=logistic)

    def with_dictionary(self, dictionary: torch.Tensor) -> FrozenDetectors:
        return FrozenDetectors(mean=self.mean, logistic=self.logistic, dictionary=dictionary)

    def score(self, residual: torch.Tensor, dictionary: torch.Tensor) -> dict[str, Any]:
        if dictionary is None:
            raise RuntimeError("J-space dictionary is required for Phase 4 scoring")
        reconstruction, support_ids, coefficients = screened_nonnegative_pursuit(
            residual.reshape(1, -1),
            dictionary,
            sparsity_k=int(self.mean["decomposition"]["sparsity_k"]),
            screen_candidates=int(self.mean["decomposition"]["screen_candidates"]),
        )
        h_j = reconstruction[0].float()
        mean_score = float(
            torch.dot(
                self.mean["d_unit"].float(), h_j - self.mean["mu_clean"].float()
            )
        )

        feature_ids = self.logistic["feature_token_ids"].numpy().astype(np.int64)
        weights = self.logistic["weights"].numpy().astype(np.float64)
        ids = support_ids[0].numpy().astype(np.int64)
        values = coefficients[0].numpy().astype(np.float64)
        positions = np.searchsorted(feature_ids, ids)
        known = (ids >= 0) & (positions < len(feature_ids))
        known_indices = np.flatnonzero(known)
        known[known_indices] &= feature_ids[positions[known_indices]] == ids[known_indices]
        logistic_score = float(self.logistic["intercept"])
        logistic_score += float(np.dot(values[known], weights[positions[known]]))
        return {
            "mean_score": mean_score,
            "mean_prediction": mean_score >= float(self.mean["threshold"]),
            "logistic_score": logistic_score,
            "logistic_prediction": logistic_score >= float(self.logistic["threshold"]),
        }
