from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..phase1.config import Phase1Config
from ..phase1.config import load_config as load_phase1_config


@dataclass(frozen=True)
class Phase3Config:
    phase1: Phase1Config
    phase1_selected_path: Path
    output_dir: Path
    penalty: str
    regularization_c: float
    solver: str
    fit_intercept: bool
    class_weight: None
    random_state: int
    max_iter: int
    tol: float

    def validate(self) -> None:
        self.phase1.validate()
        expected = {
            "penalty": "l2",
            "regularization_c": 1.0,
            "solver": "liblinear",
            "fit_intercept": True,
            "class_weight": None,
            "random_state": 42,
            "max_iter": 1000,
            "tol": 1e-4,
        }
        actual = {
            "penalty": self.penalty,
            "regularization_c": self.regularization_c,
            "solver": self.solver,
            "fit_intercept": self.fit_intercept,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }
        if actual != expected:
            raise ValueError(f"Phase 3 requires fixed logistic settings: {expected}")

    def scientific_dict(self) -> dict[str, Any]:
        return {
            "phase1_config_sha256": self.phase1.identity_hash(),
            "logistic": {
                "penalty": self.penalty,
                "C": self.regularization_c,
                "solver": self.solver,
                "fit_intercept": self.fit_intercept,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "feature_scaling": False,
            },
            "threshold_metric": "macro_balanced_accuracy",
            "threshold_tie_break": "higher_threshold",
            "prediction_rule": "score_greater_than_or_equal_to_threshold",
        }

    def identity_hash(self) -> str:
        payload = json.dumps(self.scientific_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(
    path: str | Path,
    *,
    phase1_selected_path: str | Path,
    output_dir: str | Path,
) -> Phase3Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("phase3"), dict):
        raise ValueError(f"Configuration does not contain a phase3 mapping: {config_path}")
    settings = raw["phase3"]
    config = Phase3Config(
        phase1=load_phase1_config(config_path),
        phase1_selected_path=Path(phase1_selected_path).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
        penalty=str(settings["penalty"]),
        regularization_c=float(settings["regularization_c"]),
        solver=str(settings["solver"]),
        fit_intercept=bool(settings["fit_intercept"]),
        class_weight=settings["class_weight"],
        random_state=int(settings["random_state"]),
        max_iter=int(settings["max_iter"]),
        tol=float(settings["tol"]),
    )
    config.validate()
    return config
