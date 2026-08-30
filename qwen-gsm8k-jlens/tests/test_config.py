from __future__ import annotations

from pathlib import Path

import pytest

from gsm8k_jspace.config import (
    ConfigError,
    apply_cli_overrides,
    condition_fingerprint,
    load_config,
    run_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_default_and_smoke_load():
    default = load_config(CONFIGS / "default.yaml")
    smoke = load_config(CONFIGS / "smoke.yaml")
    assert default.experiment.condition == "baseline"
    assert smoke.benchmark.subset_size == 5
    assert smoke.capture.enabled is False


def test_unknown_key_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nexperiment:\n  nope: 1\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_invalid_condition_combination(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\nexperiment:\n  condition: baseline\nintervention:\n  enabled: true\n"
    )
    with pytest.raises(ConfigError, match="intervention.enabled=false"):
        load_config(path)


def test_intervention_requires_enabled(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\nexperiment:\n  condition: intervention\nintervention:\n  enabled: false\n"
    )
    with pytest.raises(ConfigError, match="intervention.enabled=true"):
        load_config(path)


def test_layer_selector_validation(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\ncapture:\n  layers:\n    mode: explicit\n    values: []\n"
    )
    with pytest.raises(ConfigError, match="values must be non-empty"):
        load_config(path)


def test_overlay_mean_replace():
    from gsm8k_jspace.config import config_from_mapping, deep_merge, load_yaml_mapping

    merged = deep_merge(
        load_yaml_mapping(CONFIGS / "default.yaml"),
        load_yaml_mapping(CONFIGS / "experiments" / "mean-replace.yaml"),
    )
    cfg = config_from_mapping(merged)
    assert cfg.experiment.condition == "intervention"
    assert cfg.intervention.enabled is True
    assert cfg.intervention.method == "mean_replace"
    off = config_from_mapping(
        deep_merge(
            load_yaml_mapping(CONFIGS / "default.yaml"),
            load_yaml_mapping(CONFIGS / "experiments" / "capture-off.yaml"),
        )
    )
    assert off.capture.enabled is False


def test_fingerprints_stable_and_condition_sensitive():
    baseline = load_config(CONFIGS / "default.yaml")
    again = load_config(CONFIGS / "default.yaml")
    assert run_fingerprint(baseline) == run_fingerprint(again)
    no_op = apply_cli_overrides(load_config(CONFIGS / "default.yaml"), condition="no_op")
    assert run_fingerprint(baseline) == run_fingerprint(no_op)
    assert condition_fingerprint(baseline) != condition_fingerprint(no_op)


def test_m1_host_overlay_uses_mlx():
    from gsm8k_jspace.config import config_from_mapping, deep_merge, load_yaml_mapping

    merged = deep_merge(
        load_yaml_mapping(CONFIGS / "runs" / "small-smoke.yaml"),
        load_yaml_mapping(CONFIGS / "hosts" / "apple.yaml"),
    )
    cfg = config_from_mapping(merged)
    assert cfg.runtime.backend == "mlx"
    assert cfg.runtime.host_profile == "m1-max"


def test_cli_limit_override():
    cfg = apply_cli_overrides(load_config(CONFIGS / "default.yaml"), limit=3)
    assert cfg.benchmark.subset_size == 3
    assert cfg.benchmark.full_run is False


def test_cli_capture_override():
    smoke = apply_cli_overrides(load_config(CONFIGS / "smoke.yaml"), capture=True)
    assert smoke.capture.enabled is True
    default = apply_cli_overrides(load_config(CONFIGS / "default.yaml"), capture=False)
    assert default.capture.enabled is False
    unchanged = apply_cli_overrides(load_config(CONFIGS / "smoke.yaml"))
    assert unchanged.capture.enabled is False


def test_cli_capture_flags_parse():
    from gsm8k_jspace.cli import build_parser

    parser = build_parser()
    none = parser.parse_args(["run", "--config", "configs/smoke.yaml"])
    assert none.capture is None
    on = parser.parse_args(["run", "--config", "configs/smoke.yaml", "--capture"])
    assert on.capture is True
    off = parser.parse_args(["run", "--config", "configs/smoke.yaml", "--no-capture"])
    assert off.capture is False


def test_new_benchmark_configs_load():
    bipia = load_config(CONFIGS / "bipia-smoke.yaml")
    assert bipia.benchmark.name == "bipia"
    assert bipia.evaluation.parser == "bipia_asr_v1"
    agentdojo = load_config(CONFIGS / "agentdojo-smoke.yaml")
    assert agentdojo.benchmark.name == "agentdojo"
    injecagent = load_config(CONFIGS / "injecagent-smoke.yaml")
    assert injecagent.benchmark.injecagent.attack == "dh"


def test_parser_must_match_benchmark(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\n"
        "benchmark:\n  name: bipia\n"
        "evaluation:\n  parser: gsm8k_numeric_v1\n"
    )
    with pytest.raises(ConfigError, match="evaluation.parser='bipia_asr_v1'"):
        load_config(path)

