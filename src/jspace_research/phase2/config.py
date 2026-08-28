from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..phase1.config import Phase1Config
from ..phase1.config import load_config as load_phase1_config

FIXED_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FIXED_JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"


@dataclass(frozen=True)
class Phase2Config:
    phase1: Phase1Config
    phase1_selected_path: Path
    output_dir: Path
    alphas: tuple[float, ...]
    max_new_tokens: int
    do_sample: bool
    generation_batch_size: int
    judge_model: str

    @property
    def smoke(self) -> bool:
        return self.phase1.smoke_layer_count is not None

    def validate(self) -> None:
        self.phase1.validate()
        if self.alphas != FIXED_ALPHAS:
            raise ValueError(f"Phase 2 requires fixed alphas {FIXED_ALPHAS}")
        if self.max_new_tokens != 512:
            raise ValueError("Phase 2 requires max_new_tokens=512")
        if self.do_sample:
            raise ValueError("Phase 2 requires deterministic greedy generation")
        if self.generation_batch_size != 1:
            raise ValueError("Phase 2 currently requires generation_batch_size=1")
        if self.judge_model != FIXED_JUDGE_MODEL:
            raise ValueError(f"Phase 2 requires judge_model={FIXED_JUDGE_MODEL}")

    def scientific_dict(self) -> dict[str, Any]:
        return {
            "phase1_config_sha256": self.phase1.identity_hash(),
            "alphas": list(self.alphas),
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "generation_batch_size": self.generation_batch_size,
            "judge_model": self.judge_model,
        }

    def identity_hash(self) -> str:
        payload = json.dumps(self.scientific_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(
    path: str | Path,
    *,
    phase1_selected_path: str | Path,
    output_dir: str | Path,
) -> Phase2Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("phase2"), dict):
        raise ValueError(f"Configuration does not contain a phase2 mapping: {config_path}")
    phase2_raw = raw["phase2"]
    phase1 = load_phase1_config(config_path)
    config = Phase2Config(
        phase1=phase1,
        phase1_selected_path=Path(phase1_selected_path).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
        alphas=tuple(float(value) for value in phase2_raw["alphas"]),
        max_new_tokens=int(phase2_raw["max_new_tokens"]),
        do_sample=bool(phase2_raw["do_sample"]),
        generation_batch_size=int(phase2_raw["generation_batch_size"]),
        judge_model=str(phase2_raw["judge_model"]),
    )
    config.validate()
    return config
