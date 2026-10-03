from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..phase1.config import Phase1Config
from ..phase1.config import load_config as load_phase1_config
from ..phase4.config import (
    AGENTDOJO_REVISION,
    AGENTDOJO_SUITES,
    INJECAGENT_REVISION,
    Phase4Config,
)


@dataclass(frozen=True)
class RawBaselineConfig:
    phase1: Phase1Config
    phase1_selected_path: Path
    jspace_phase3_dir: Path
    jspace_phase4_dir: Path
    agentdojo_root: Path
    injecagent_root: Path
    output_dir: Path
    agentdojo_revision: str
    agentdojo_version: str
    agentdojo_suites: tuple[str, ...]
    agentdojo_attack: str
    agentdojo_defense: None
    injecagent_revision: str
    injecagent_setting: str
    injecagent_prompt_type: str
    max_new_tokens: int
    penalty: str
    regularization_c: float
    solver: str
    fit_intercept: bool
    class_weight: None
    random_state: int
    max_iter: int
    tol: float
    bootstrap_replicates: int = 2000

    @property
    def smoke(self) -> bool:
        return self.phase1.smoke_layer_count is not None

    def validate(self, *, require_roots: bool = False) -> None:
        self.phase1.validate()
        expected = {
            "agentdojo_revision": AGENTDOJO_REVISION,
            "agentdojo_version": "v1.2.2",
            "agentdojo_suites": AGENTDOJO_SUITES,
            "agentdojo_attack": "important_instructions",
            "agentdojo_defense": None,
            "injecagent_revision": INJECAGENT_REVISION,
            "injecagent_setting": "base",
            "injecagent_prompt_type": "InjecAgent",
            "max_new_tokens": 512,
            "penalty": "l2",
            "regularization_c": 1.0,
            "solver": "liblinear",
            "fit_intercept": True,
            "class_weight": None,
            "random_state": 42,
            "max_iter": 1000,
            "tol": 1e-4,
            "bootstrap_replicates": 2000,
        }
        actual = {key: getattr(self, key) for key in expected}
        if actual != expected:
            raise ValueError(f"Raw baseline requires fixed settings: {expected}")
        for name, path in (
            ("Phase 1 selection", self.phase1_selected_path),
            ("Phase 3 directory", self.jspace_phase3_dir),
            ("Phase 4 directory", self.jspace_phase4_dir),
        ):
            if require_roots and not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
        if require_roots:
            for name, root in (
                ("AgentDojo", self.agentdojo_root),
                ("InjecAgent", self.injecagent_root),
            ):
                if not root.is_dir():
                    raise FileNotFoundError(f"{name} checkout not found: {root}")

    def scientific_dict(self) -> dict[str, Any]:
        return {
            "phase1_config_sha256": self.phase1.identity_hash(),
            "representation": "raw_residual_stream",
            "layer_policy": "frozen_phase1_selected_layer",
            "decision_points": "identical_to_phase3_and_phase4",
            "feature_scaling": False,
            "logistic": {
                "penalty": self.penalty,
                "C": self.regularization_c,
                "solver": self.solver,
                "fit_intercept": self.fit_intercept,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                "max_iter": self.max_iter,
                "tol": self.tol,
            },
            "threshold_metric": "macro_balanced_accuracy",
            "threshold_tie_break": "higher_threshold",
            "bootstrap": {
                "seed": 42,
                "replicates": self.bootstrap_replicates,
                "interval": 0.95,
            },
            "benchmarks": {
                "agentdojo_revision": self.agentdojo_revision,
                "agentdojo_version": self.agentdojo_version,
                "agentdojo_suites": list(self.agentdojo_suites),
                "agentdojo_attack": self.agentdojo_attack,
                "agentdojo_defense": self.agentdojo_defense,
                "injecagent_revision": self.injecagent_revision,
                "injecagent_setting": self.injecagent_setting,
                "injecagent_prompt_type": self.injecagent_prompt_type,
            },
            "max_new_tokens": self.max_new_tokens,
            "smoke": self.smoke,
        }

    def identity_hash(self) -> str:
        payload = json.dumps(self.scientific_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def phase4_config(self, output_dir: Path | None = None) -> Phase4Config:
        return Phase4Config(
            phase1=self.phase1,
            phase1_selected_path=self.phase1_selected_path,
            phase3_dir=self.jspace_phase3_dir,
            bipia_root=self.phase1.data.bipia_root,
            agentdojo_root=self.agentdojo_root,
            injecagent_root=self.injecagent_root,
            output_dir=output_dir or self.output_dir,
            agentdojo_revision=self.agentdojo_revision,
            agentdojo_version=self.agentdojo_version,
            agentdojo_suites=self.agentdojo_suites,
            agentdojo_attack=self.agentdojo_attack,
            agentdojo_defense=self.agentdojo_defense,
            injecagent_revision=self.injecagent_revision,
            injecagent_setting=self.injecagent_setting,
            injecagent_prompt_type=self.injecagent_prompt_type,
            max_new_tokens=self.max_new_tokens,
            judge_model="openai/gpt-4.1-mini",
        )


def load_config(
    path: str | Path,
    *,
    phase1_selected_path: str | Path,
    jspace_phase3_dir: str | Path,
    jspace_phase4_dir: str | Path,
    agentdojo_root: str | Path,
    injecagent_root: str | Path,
    output_dir: str | Path,
) -> RawBaselineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    phase3 = raw.get("phase3")
    phase4 = raw.get("phase4")
    if not isinstance(phase3, dict) or not isinstance(phase4, dict):
        raise ValueError("Raw baseline requires phase3 and phase4 configuration mappings")
    config = RawBaselineConfig(
        phase1=load_phase1_config(config_path),
        phase1_selected_path=Path(phase1_selected_path).expanduser().resolve(),
        jspace_phase3_dir=Path(jspace_phase3_dir).expanduser().resolve(),
        jspace_phase4_dir=Path(jspace_phase4_dir).expanduser().resolve(),
        agentdojo_root=Path(agentdojo_root).expanduser().resolve(),
        injecagent_root=Path(injecagent_root).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
        agentdojo_revision=str(phase4["agentdojo_revision"]),
        agentdojo_version=str(phase4["agentdojo_version"]),
        agentdojo_suites=tuple(phase4["agentdojo_suites"]),
        agentdojo_attack=str(phase4["agentdojo_attack"]),
        agentdojo_defense=phase4["agentdojo_defense"],
        injecagent_revision=str(phase4["injecagent_revision"]),
        injecagent_setting=str(phase4["injecagent_setting"]),
        injecagent_prompt_type=str(phase4["injecagent_prompt_type"]),
        max_new_tokens=int(phase4["max_new_tokens"]),
        penalty=str(phase3["penalty"]),
        regularization_c=float(phase3["regularization_c"]),
        solver=str(phase3["solver"]),
        fit_intercept=bool(phase3["fit_intercept"]),
        class_weight=phase3["class_weight"],
        random_state=int(phase3["random_state"]),
        max_iter=int(phase3["max_iter"]),
        tol=float(phase3["tol"]),
    )
    config.validate()
    return config
