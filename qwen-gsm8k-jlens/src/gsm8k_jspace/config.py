"""Typed YAML configuration, validation, and fingerprints."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from gsm8k_jspace import SCHEMA_VERSION

KNOWN_CONDITIONS = ("baseline", "no_op", "intervention")
KNOWN_LAYER_MODES = ("late", "all_fitted", "explicit", "range")
KNOWN_TOKEN_MODES = (
    "prompt_last",
    "generated_last",
    "all_generated",
    "generated_stride",
    "word_end",
    "explicit",
    "full_sequence",
)
KNOWN_INTERVENTION_METHODS = ("mean_replace", "none")
KNOWN_FEATURE_MODES = ("top_abs", "explicit", "random_matched")


class ConfigError(ValueError):
    """Invalid or contradictory configuration."""


def _require_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {allowed}, got {value!r}")


@dataclass
class LayerSelector:
    mode: str = "late"
    values: list[int] = field(default_factory=list)
    start: int | None = None
    stop: int | None = None
    stride: int = 1

    def validate(self, prefix: str) -> None:
        _require_choice(f"{prefix}.mode", self.mode, KNOWN_LAYER_MODES)
        if self.stride < 1:
            raise ConfigError(f"{prefix}.stride must be >= 1")
        if self.mode == "explicit" and not self.values:
            raise ConfigError(f"{prefix}.values must be non-empty when mode=explicit")
        if self.mode == "range" and (self.start is None or self.stop is None):
            raise ConfigError(f"{prefix}.start and .stop are required when mode=range")


@dataclass
class TokenSelector:
    mode: str = "all_generated"
    stride: int = 1
    positions: list[int] = field(default_factory=list)
    include_prompt: bool = False

    def validate(self, prefix: str) -> None:
        _require_choice(f"{prefix}.mode", self.mode, KNOWN_TOKEN_MODES)
        if self.stride < 1:
            raise ConfigError(f"{prefix}.stride must be >= 1")
        if self.mode == "explicit" and not self.positions:
            raise ConfigError(f"{prefix}.positions must be non-empty when mode=explicit")
        if self.mode == "generated_stride" and self.stride < 1:
            raise ConfigError(f"{prefix}.stride must be >= 1")


@dataclass
class CaptureFields:
    hidden_norm: bool = True
    jspace_norm: bool = True
    top_jspace_tokens: bool = True
    top_logit_tokens: bool = True
    top_model_tokens: bool = True
    hidden_vector: bool = False
    jspace_vector: bool = False
    intervention_delta_norm: bool = True


@dataclass
class ExperimentSection:
    name: str = "gsm8k-qwen35-9b"
    condition: str = "baseline"
    run_tier: str = "small"
    tags: list[str] = field(default_factory=lambda: ["gsm8k", "jspace"])
    notes: str | None = None


@dataclass
class ModelSection:
    name: str = "Qwen/Qwen3.5-9B-Base"
    revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "auto"
    device_map: str = "auto"
    offload_folder: str | None = None
    attention_implementation: str | None = None
    trust_remote_code: bool = False
    mlx_repo: str | None = None


@dataclass
class JLensSection:
    required: bool = True
    source: str = "hub"
    repo: str = "neuronpedia/jacobian-lens"
    revision: str = "qwen-n1000"
    file: str = (
        "qwen3.5-9b-pt/jlens/Salesforce-wikitext/"
        "Qwen3.5-9B-Base_jacobian_lens.pt"
    )
    local_path: str | None = None


@dataclass
class BenchmarkSection:
    test_type: str = "full_answer"
    dataset: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "test"
    full_run: bool = False
    subset_size: int = 10
    selection: str = "first"
    selection_seed: int = 0


@dataclass
class PromptSection:
    template: str = "zero_shot_cot_v1"
    answer_marker: str = "####"
    few_shot_examples: int = 0
    context_overflow: str = "error"
    final_token_capture: str = "replay"
    use_chat_template: bool = False


@dataclass
class GenerationSection:
    mode: str = "greedy"
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    stop_strings: list[str] = field(default_factory=list)


@dataclass
class CaptureSection:
    enabled: bool = True
    phases: list[str] = field(default_factory=lambda: ["post_layer"])
    layers: LayerSelector = field(default_factory=LayerSelector)
    tokens: TokenSelector = field(default_factory=TokenSelector)
    fields: CaptureFields = field(default_factory=CaptureFields)
    top_k_tokens: int = 20
    vector_dtype: str = "float16"
    capture_pre_intervention: bool = True
    capture_post_intervention: bool = False


@dataclass
class FeatureSection:
    mode: str = "top_abs"
    top_k: int = 50
    indices: list[int] = field(default_factory=list)
    random_seed: int = 0


@dataclass
class InterventionSection:
    enabled: bool = False
    method: str = "mean_replace"
    layers: LayerSelector = field(default_factory=LayerSelector)
    tokens: TokenSelector = field(
        default_factory=lambda: TokenSelector(mode="all_generated", include_prompt=True)
    )
    features: FeatureSection = field(default_factory=FeatureSection)
    strength: float = 0.05
    reference: str = "task_running_mean"


@dataclass
class EvaluationSection:
    parser: str = "gsm8k_numeric_v1"
    prefer_answer_marker: bool = True
    allow_last_number_fallback: bool = True
    numeric_tolerance: float = 0.0


@dataclass
class OutputsSection:
    root_dir: str = "outputs/gsm8k"
    run_id: str | None = None
    on_existing: str = "error"
    completion_format: str = "jsonl"
    capture_format: str = "jsonl_gzip"
    save_prompts: bool = True
    save_generated_token_ids: bool = True
    flush_every_examples: int = 1


@dataclass
class VisualizationSection:
    enabled: bool = True
    notebooks: list[str] = field(
        default_factory=lambda: ["run-overview", "gsm8k-accuracy"]
    )
    export_formats: list[str] = field(default_factory=lambda: ["ipynb", "html"])
    figure_formats: list[str] = field(default_factory=lambda: ["png", "svg"])
    save_backing_tables: bool = True
    max_examples_in_tables: int = 100
    max_capture_rows_in_memory: int = 250_000


@dataclass
class PinvSection:
    compute_device: str = "cpu"
    cache_device: str = "cpu"
    cache_dir: str = ".cache/jspace/pinv"
    preload: bool = False


@dataclass
class RuntimeSection:
    host_profile: str = "auto"
    backend: str = "auto"
    device: str = "auto"
    linear_algebra_device: str = "auto"
    compatibility_mode: str = "strict"
    memory_preflight: bool = True
    minimum_free_memory_gb: float = 2.0
    pinv: PinvSection = field(default_factory=PinvSection)
    offline: bool = False
    deterministic_algorithms: bool = False
    log_level: str = "INFO"


@dataclass
class AppConfig:
    schema_version: int = SCHEMA_VERSION
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    model: ModelSection = field(default_factory=ModelSection)
    jlens: JLensSection = field(default_factory=JLensSection)
    benchmark: BenchmarkSection = field(default_factory=BenchmarkSection)
    prompt: PromptSection = field(default_factory=PromptSection)
    generation: GenerationSection = field(default_factory=GenerationSection)
    capture: CaptureSection = field(default_factory=CaptureSection)
    intervention: InterventionSection = field(default_factory=InterventionSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    outputs: OutputsSection = field(default_factory=OutputsSection)
    visualization: VisualizationSection = field(default_factory=VisualizationSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigError(
                f"unsupported schema_version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        _require_choice(
            "experiment.condition", self.experiment.condition, KNOWN_CONDITIONS
        )
        _require_choice(
            "experiment.run_tier",
            self.experiment.run_tier,
            ("small", "medium", "large"),
        )
        _require_choice(
            "benchmark.test_type",
            self.benchmark.test_type,
            ("full_answer", "gold_next_token"),
        )
        _require_choice(
            "benchmark.selection", self.benchmark.selection, ("first", "shuffled")
        )
        _require_choice("generation.mode", self.generation.mode, ("greedy", "sample"))
        _require_choice(
            "prompt.context_overflow",
            self.prompt.context_overflow,
            ("error", "reduce_few_shot"),
        )
        _require_choice(
            "prompt.final_token_capture",
            self.prompt.final_token_capture,
            ("replay", "predicting_state"),
        )
        _require_choice(
            "jlens.source", self.jlens.source, ("hub", "local", "identity")
        )
        _require_choice(
            "outputs.on_existing", self.outputs.on_existing, ("error", "resume")
        )
        _require_choice(
            "runtime.backend",
            self.runtime.backend,
            ("auto", "mlx", "mps", "cuda", "rocm", "cpu"),
        )
        _require_choice(
            "runtime.compatibility_mode",
            self.runtime.compatibility_mode,
            ("strict", "compatible"),
        )
        _require_choice(
            "runtime.host_profile",
            self.runtime.host_profile,
            ("auto", "m1-max", "radeon-8060s", "nvidia-datacenter", "cpu"),
        )
        _require_choice(
            "intervention.method",
            self.intervention.method,
            KNOWN_INTERVENTION_METHODS,
        )
        _require_choice(
            "intervention.features.mode",
            self.intervention.features.mode,
            KNOWN_FEATURE_MODES,
        )
        if not 0.0 <= float(self.intervention.strength) <= 1.0:
            raise ConfigError("intervention.strength must be in [0, 1]")
        if self.benchmark.subset_size < 1 and not self.benchmark.full_run:
            raise ConfigError("benchmark.subset_size must be >= 1 unless full_run")
        if self.generation.mode == "greedy":
            self.generation.do_sample = False
        condition = self.experiment.condition
        enabled = self.intervention.enabled
        if condition in ("baseline", "no_op") and enabled:
            raise ConfigError(
                f"experiment.condition={condition!r} requires intervention.enabled=false"
            )
        if condition == "intervention" and not enabled:
            raise ConfigError(
                "experiment.condition='intervention' requires intervention.enabled=true"
            )
        if self.jlens.source == "local" and not self.jlens.local_path:
            raise ConfigError("jlens.local_path is required when source=local")
        self.capture.layers.validate("capture.layers")
        self.capture.tokens.validate("capture.tokens")
        self.intervention.layers.validate("intervention.layers")
        self.intervention.tokens.validate("intervention.tokens")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nested_type(cls: type, name: str) -> type | None:
    hint = get_type_hints(cls).get(name)
    if isinstance(hint, type) and is_dataclass(hint):
        return hint
    return None


def _fill_dataclass(cls: type, data: dict[str, Any], prefix: str = "") -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{prefix or cls.__name__} must be a mapping")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        where = prefix or cls.__name__
        raise ConfigError(f"unknown key(s) under {where}: {unknown}")
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in data:
            continue
        value = data[item.name]
        nested = _nested_type(cls, item.name)
        path = f"{prefix}.{item.name}" if prefix else item.name
        if nested is not None:
            kwargs[item.name] = _fill_dataclass(nested, value, path)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return loaded


def config_from_mapping(data: dict[str, Any]) -> AppConfig:
    cfg = _fill_dataclass(AppConfig, data)
    cfg.validate()
    return cfg


def load_config(
    path: str | Path,
    *,
    host_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    data = load_yaml_mapping(path)
    if host_path is not None:
        data = deep_merge(data, load_yaml_mapping(host_path))
    if overrides:
        data = deep_merge(data, overrides)
    return config_from_mapping(data)


def apply_cli_overrides(
    cfg: AppConfig,
    *,
    limit: int | None = None,
    run_id: str | None = None,
    resume: bool = False,
    condition: str | None = None,
    capture: bool | None = None,
) -> AppConfig:
    if limit is not None:
        cfg.benchmark.subset_size = int(limit)
        cfg.benchmark.full_run = False
    if run_id is not None:
        cfg.outputs.run_id = run_id
    if resume:
        cfg.outputs.on_existing = "resume"
    if condition is not None:
        cfg.experiment.condition = condition
        cfg.intervention.enabled = condition == "intervention"
    if capture is not None:
        cfg.capture.enabled = bool(capture)
    cfg.validate()
    return cfg


def _canonical(obj: Any) -> Any:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {str(key): _canonical(obj[key]) for key in sorted(obj)}
    if isinstance(obj, list):
        return [_canonical(item) for item in obj]
    return obj


def fingerprint(payload: Any) -> str:
    encoded = json.dumps(_canonical(payload), separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def run_fingerprint(cfg: AppConfig) -> str:
    payload = {
        "model": cfg.model,
        "jlens": cfg.jlens,
        "benchmark": cfg.benchmark,
        "prompt": cfg.prompt,
        "generation": cfg.generation,
        "evaluation": cfg.evaluation,
    }
    return fingerprint(payload)


def condition_fingerprint(cfg: AppConfig) -> str:
    payload = {
        "condition": cfg.experiment.condition,
        "capture": cfg.capture,
        "intervention": cfg.intervention,
    }
    return fingerprint(payload)


def dump_resolved_yaml(cfg: AppConfig) -> str:
    return yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True)
