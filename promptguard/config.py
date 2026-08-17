"""Typed configuration for the promptguard research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3.5-4B"
    layer_indices: list[int] = field(default_factory=lambda: list(range(3, 32, 4)))
    hidden_dim: int | None = 2560
    layer_path: str | None = None
    max_length: int = 512
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    trust_remote_code: bool = False


@dataclass
class ProbeConfig:
    pooling: str = "last_token"
    aggregate: str = "mean_logits"
    threshold: float = 0.5
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 16
    weight_decay: float = 1e-4
    layer_loss_weight: float = 0.25


@dataclass
class InterventionConfig:
    mode: str = "circuit_breaker"
    threshold: float = 0.5
    beta: float = 1.0
    refusal: str = "I cannot comply with that request."
    positions: str = "all"


@dataclass
class AttackerConfig:
    alpha: float = 0.2
    gamma: float = 0.9
    attempts_per_prompt: int = 4
    sandbox_tau: float = 0.25
    mutation_validation_samples: int = 8
    dedup_similarity: float = 0.92
    max_reflections_per_round: int = 4


@dataclass
class GuardConfig:
    lambda_raw: float = 0.5
    confidence_gate: float = 0.95
    replay_previous_rounds: bool = True
    review_queue_path: str = "outputs/review_queue.csv"


@dataclass
class EvolutionConfig:
    rounds: int = 10
    seed: int = 7
    output_dir: str = "outputs/evolution"
    baseline_prompt: str = "Classify and safely respond to the following user text:"
    checkpoint_dir: str = "outputs/checkpoints"


@dataclass
class DataConfig:
    benign_prompts: list[str] = field(default_factory=list)
    malicious_prompts: list[str] = field(default_factory=list)
    heldout_malicious_prompts: list[str] = field(default_factory=list)
    use_benchmarks_for_evolution: bool = False


@dataclass
class OlaresConfig:
    base_url: str = "https://a5be22681.dav50505.olares.com"
    api_key_env: str = "OLARES_API_KEY"
    default_api_key: str = "ollama"
    embed_model: str = "nomic-embed-text"
    fast_model: str = "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:Q8_0"
    reason_model: str = "glm-fixed"
    code_model: str = "qwen3-coder:30b"
    fallback_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fallback_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    timeout_seconds: float = 45.0
    force_local_fallback: bool = False


@dataclass
class BenchmarkConfig:
    harmbench_url: str = (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
        "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    )
    advbench_url: str = (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
        "main/data/advbench/harmful_behaviors.csv"
    )
    cache_dir: str = "data/benchmarks"
    harmbench_subset_size: int | str = 50
    advbench_subset_size: int | str = 50
    heldout_fraction: float = 0.2
    heldout_categories: list[str] = field(default_factory=list)
    seed: int = 7


@dataclass
class BaselineConfig:
    enabled: list[str] = field(default_factory=lambda: ["renellm"])
    query_budget: int = 8
    tap_candidates_per_iteration: int = 3
    tap_beam_width: int = 2
    tap_iterations: int = 3
    output_dir: str = "outputs/benchmarks"


@dataclass
class MetricsConfig:
    enable_harmfulness_judge: bool = False
    judge_max_tokens: int = 128


@dataclass
class ResearchConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    olares: OlaresConfig = field(default_factory=OlaresConfig)
    benchmarks: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


def _construct(cls: type, values: dict[str, Any] | None):
    allowed = cls.__dataclass_fields__
    values = values or {}
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ResearchConfig:
    """Load and validate a YAML configuration file."""

    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    sections = {
        "model": ModelConfig,
        "probe": ProbeConfig,
        "intervention": InterventionConfig,
        "attacker": AttackerConfig,
        "guard": GuardConfig,
        "evolution": EvolutionConfig,
        "data": DataConfig,
        "olares": OlaresConfig,
        "benchmarks": BenchmarkConfig,
        "baselines": BaselineConfig,
        "metrics": MetricsConfig,
    }
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    return ResearchConfig(
        **{name: _construct(cls, raw.get(name)) for name, cls in sections.items()}
    )
