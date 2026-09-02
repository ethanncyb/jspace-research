from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..phase1.config import Phase1Config
from ..phase1.config import load_config as load_phase1_config

AGENTDOJO_REVISION = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
INJECAGENT_REVISION = "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"
AGENTDOJO_SUITES = ("banking", "slack", "travel", "workspace")
BIPIA_ATTACKS_PER_TASK = 250


@dataclass(frozen=True)
class Phase4Config:
    phase1: Phase1Config
    phase1_selected_path: Path
    phase3_dir: Path
    bipia_root: Path
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
    judge_model: str

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
        }
        actual = {key: getattr(self, key) for key in expected}
        if actual != expected:
            raise ValueError(f"Phase 4 requires fixed benchmark settings: {expected}")
        if self.judge_model != "openai/gpt-4.1-mini":
            raise ValueError("Phase 4 requires the frozen Phase 2 OpenRouter judge model")
        if require_roots:
            for name, root in (
                ("BIPIA", self.bipia_root.parent),
                ("AgentDojo", self.agentdojo_root),
                ("InjecAgent", self.injecagent_root),
            ):
                if not root.is_dir():
                    raise FileNotFoundError(f"{name} checkout not found: {root}")

    def scientific_dict(self) -> dict[str, Any]:
        return {
            "phase1_config_sha256": self.phase1.identity_hash(),
            "benchmarks": {
                "bipia_revision": self.phase1.dependencies.bipia_revision,
                "bipia_split": "official_test",
                "bipia_sampling": {
                    "seed": 42,
                    "attacks_per_task": BIPIA_ATTACKS_PER_TASK,
                    "stratification": "attack_category_x_insertion_position",
                    "test_variants": [0, 1, 2, 3, 4],
                    "positions": ["start", "middle", "end"],
                    "clean_controls": "one_per_unique_source_context",
                },
                "bipia_metric_weighting": "context_matched_class_balance",
                "agentdojo_revision": self.agentdojo_revision,
                "agentdojo_version": self.agentdojo_version,
                "agentdojo_suites": list(self.agentdojo_suites),
                "agentdojo_attack": self.agentdojo_attack,
                "agentdojo_defense": self.agentdojo_defense,
                "agentdojo_tool_output_format": "native_default_yaml",
                "injecagent_revision": self.injecagent_revision,
                "injecagent_setting": self.injecagent_setting,
                "injecagent_prompt_type": self.injecagent_prompt_type,
            },
            "max_new_tokens": self.max_new_tokens,
            "judge_model": self.judge_model,
            "smoke": self.smoke,
        }

    def identity_hash(self) -> str:
        payload = json.dumps(self.scientific_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_config(
    path: str | Path,
    *,
    phase1_selected_path: str | Path,
    phase3_dir: str | Path,
    bipia_root: str | Path,
    agentdojo_root: str | Path,
    injecagent_root: str | Path,
    output_dir: str | Path,
) -> Phase4Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("phase4"), dict):
        raise ValueError(f"Configuration does not contain a phase4 mapping: {config_path}")
    settings = raw["phase4"]
    phase2 = raw.get("phase2", {})
    config = Phase4Config(
        phase1=load_phase1_config(config_path, bipia_root=bipia_root),
        phase1_selected_path=Path(phase1_selected_path).expanduser().resolve(),
        phase3_dir=Path(phase3_dir).expanduser().resolve(),
        bipia_root=Path(bipia_root).expanduser().resolve(),
        agentdojo_root=Path(agentdojo_root).expanduser().resolve(),
        injecagent_root=Path(injecagent_root).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
        agentdojo_revision=str(settings["agentdojo_revision"]),
        agentdojo_version=str(settings["agentdojo_version"]),
        agentdojo_suites=tuple(settings["agentdojo_suites"]),
        agentdojo_attack=str(settings["agentdojo_attack"]),
        agentdojo_defense=settings["agentdojo_defense"],
        injecagent_revision=str(settings["injecagent_revision"]),
        injecagent_setting=str(settings["injecagent_setting"]),
        injecagent_prompt_type=str(settings["injecagent_prompt_type"]),
        max_new_tokens=int(settings["max_new_tokens"]),
        judge_model=str(phase2["judge_model"]),
    )
    config.validate()
    return config
