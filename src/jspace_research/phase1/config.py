from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_TASKS = ("email", "qa", "table", "abstract", "code")
EXPECTED_JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
EXPECTED_BIPIA_REVISION = "a004b69ec0dd446e0afd461d98cb5e96e120a5d0"


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str
    precision: str = "bfloat16"


@dataclass(frozen=True)
class LensConfig:
    repository: str
    revision: str
    filename: str
    sha256: str


@dataclass(frozen=True)
class DependencyConfig:
    jacobian_lens_revision: str
    bipia_revision: str


@dataclass(frozen=True)
class DataConfig:
    bipia_root: Path
    webqa_train_path: Path | None = None
    summarization_train_path: Path | None = None


@dataclass(frozen=True)
class Phase1Config:
    model: ModelConfig
    lens: LensConfig
    dependencies: DependencyConfig
    data: DataConfig
    output_dir: Path
    seed: int
    tasks: tuple[str, ...]
    train_pairs_per_task: int
    validation_pairs_per_task: int
    max_input_tokens: int
    token_match_tolerance: int
    sparsity_k: int
    screen_candidates: int
    decomposition_batch_size: int
    dictionary_chunk_size: int
    smoke_layer_count: int | None = None

    def validate(self, *, require_data_files: bool = False) -> None:
        unknown = sorted(set(self.tasks) - set(SUPPORTED_TASKS))
        if unknown:
            raise ValueError(f"Unsupported BIPIA tasks: {unknown}")
        if not self.tasks:
            raise ValueError("At least one BIPIA task is required")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("BIPIA tasks must be unique")
        if self.train_pairs_per_task <= 0 or self.validation_pairs_per_task <= 0:
            raise ValueError("Pair counts must be positive")
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if self.token_match_tolerance < 0:
            raise ValueError("token_match_tolerance cannot be negative")
        if not 0 < self.sparsity_k <= self.screen_candidates:
            raise ValueError("Require 0 < sparsity_k <= screen_candidates")
        if self.decomposition_batch_size <= 0 or self.dictionary_chunk_size <= 0:
            raise ValueError("Batch and dictionary chunk sizes must be positive")
        if self.smoke_layer_count is not None and self.smoke_layer_count <= 0:
            raise ValueError("smoke_layer_count must be positive when set")
        if self.model.precision != "bfloat16":
            raise ValueError("Phase 1 is specified to use bfloat16 model precision")
        if len(self.lens.sha256) != 64:
            raise ValueError("lens.sha256 must be a 64-character SHA-256 digest")
        if self.dependencies.jacobian_lens_revision != EXPECTED_JLENS_REVISION:
            raise ValueError(
                f"The Jacobian-lens dependency must be pinned to {EXPECTED_JLENS_REVISION}"
            )
        if self.dependencies.bipia_revision != EXPECTED_BIPIA_REVISION:
            raise ValueError(f"BIPIA must be pinned to {EXPECTED_BIPIA_REVISION}")
        if require_data_files:
            if not self.data.bipia_root.is_dir():
                raise FileNotFoundError(f"BIPIA benchmark root not found: {self.data.bipia_root}")
            if "qa" in self.tasks and not _is_file(self.data.webqa_train_path):
                raise FileNotFoundError(
                    "WebQA requires a researcher-provided BIPIA-format train.jsonl"
                )
            if "abstract" in self.tasks and not _is_file(self.data.summarization_train_path):
                raise FileNotFoundError(
                    "Summarization requires a researcher-provided BIPIA-format train.jsonl"
                )

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def identity_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_file(path: Path | None) -> bool:
    return path is not None and path.is_file()


def _optional_path(value: Any) -> Path | None:
    return None if value in (None, "") else Path(value).expanduser().resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def load_config(
    path: str | Path,
    *,
    bipia_root: str | Path | None = None,
    webqa_train_path: str | Path | None = None,
    summarization_train_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Phase1Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    data_raw = dict(raw["data"])
    if bipia_root is not None:
        data_raw["bipia_root"] = str(bipia_root)
    if webqa_train_path is not None:
        data_raw["webqa_train_path"] = str(webqa_train_path)
    if summarization_train_path is not None:
        data_raw["summarization_train_path"] = str(summarization_train_path)

    config = Phase1Config(
        model=ModelConfig(**raw["model"]),
        lens=LensConfig(**raw["lens"]),
        dependencies=DependencyConfig(**raw["dependencies"]),
        data=DataConfig(
            bipia_root=Path(data_raw["bipia_root"]).expanduser().resolve(),
            webqa_train_path=_optional_path(data_raw.get("webqa_train_path")),
            summarization_train_path=_optional_path(data_raw.get("summarization_train_path")),
        ),
        output_dir=Path(output_dir or raw["output_dir"]).expanduser().resolve(),
        seed=int(raw["seed"]),
        tasks=tuple(raw["tasks"]),
        train_pairs_per_task=int(raw["train_pairs_per_task"]),
        validation_pairs_per_task=int(raw["validation_pairs_per_task"]),
        max_input_tokens=int(raw["max_input_tokens"]),
        token_match_tolerance=int(raw["token_match_tolerance"]),
        sparsity_k=int(raw["sparsity_k"]),
        screen_candidates=int(raw["screen_candidates"]),
        decomposition_batch_size=int(raw["decomposition_batch_size"]),
        dictionary_chunk_size=int(raw["dictionary_chunk_size"]),
        smoke_layer_count=(
            None if raw.get("smoke_layer_count") is None else int(raw["smoke_layer_count"])
        ),
    )
    config.validate()
    return config
