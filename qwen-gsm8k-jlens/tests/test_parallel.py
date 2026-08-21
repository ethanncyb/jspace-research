from __future__ import annotations

from pathlib import Path

import pytest

from gsm8k_jspace.artifacts.writer import write_jsonl
from gsm8k_jspace.config import (
    ConfigError,
    apply_cli_overrides,
    config_from_mapping,
    load_config,
    parse_gpus_cli,
)
from gsm8k_jspace.runner.parallel import (
    merge_jsonl_by_example_id,
    merge_parallel_shards,
    partition_examples,
)
from gsm8k_jspace.types import GSM8KExample

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _example(index: int) -> GSM8KExample:
    return GSM8KExample(
        example_id=f"gsm8k_test_{index:06d}",
        source_index=index,
        question=f"q{index}",
        gold_rationale=f"#### {index}",
        gold_answer=str(index),
        question_sha256=f"sha{index}",
    )


def test_parse_gpus_cli():
    assert parse_gpus_cli(None) is None
    assert parse_gpus_cli("") == []
    assert parse_gpus_cli("0,2,3") == [0, 2, 3]
    with pytest.raises(ConfigError, match="duplicates"):
        parse_gpus_cli("1,1")


def test_runtime_gpus_validation(tmp_path: Path):
    base = load_config(CONFIGS / "default.yaml").to_dict()
    base["runtime"]["gpus"] = [0, -1]
    with pytest.raises(ConfigError, match=">= 0"):
        config_from_mapping(base)
    base["runtime"]["gpus"] = [0, 0]
    with pytest.raises(ConfigError, match="duplicates"):
        config_from_mapping(base)
    base["runtime"]["gpus"] = [0, 1]
    base["runtime"]["parallel"] = True
    base["runtime"]["backend"] = "cpu"
    with pytest.raises(ConfigError, match="parallel=true requires"):
        config_from_mapping(base)


def test_cli_gpus_enables_parallel():
    cfg = apply_cli_overrides(load_config(CONFIGS / "default.yaml"), gpus=[0, 1])
    assert cfg.runtime.gpus == [0, 1]
    assert cfg.runtime.parallel is True
    single = apply_cli_overrides(load_config(CONFIGS / "default.yaml"), gpus=[2])
    assert single.runtime.gpus == [2]
    assert single.runtime.parallel is False


def test_partition_examples_round_robin():
    examples = [_example(i) for i in range(5)]
    shards = partition_examples(examples, 2)
    assert [ex.example_id for ex in shards[0]] == [
        "gsm8k_test_000000",
        "gsm8k_test_000002",
        "gsm8k_test_000004",
    ]
    assert [ex.example_id for ex in shards[1]] == [
        "gsm8k_test_000001",
        "gsm8k_test_000003",
    ]
    again = partition_examples(examples, 2)
    assert [ex.example_id for ex in again[0]] == [ex.example_id for ex in shards[0]]


def test_merge_jsonl_prefers_first_and_keeps_order(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    out = tmp_path / "out.jsonl"
    write_jsonl(
        a,
        [
            {"example_id": "e0", "value": "a0"},
            {"example_id": "e1", "value": "a1"},
        ],
    )
    write_jsonl(
        b,
        [
            {"example_id": "e1", "value": "b1"},
            {"example_id": "e2", "value": "b2"},
        ],
    )
    count = merge_jsonl_by_example_id([a, b], out)
    assert count == 3
    rows = [line for line in out.read_text().splitlines() if line]
    assert '"e0"' in rows[0] and '"a0"' in rows[0]
    assert '"e1"' in rows[1] and '"a1"' in rows[1]
    assert '"e2"' in rows[2] and '"b2"' in rows[2]


def test_merge_parallel_shards(tmp_path: Path):
    (tmp_path / "captures").mkdir()
    write_jsonl(
        tmp_path / "completions.jsonl",
        [{"example_id": "e0", "source_index": 0}],
    )
    write_jsonl(
        tmp_path / "completions.shard-0.jsonl",
        [{"example_id": "e1", "source_index": 1}],
    )
    write_jsonl(
        tmp_path / "completions.shard-1.jsonl",
        [{"example_id": "e2", "source_index": 2}],
    )
    write_jsonl(
        tmp_path / "captures" / "index.shard-0.jsonl",
        [{"example_id": "e1", "path": "captures/e1.jsonl.gz"}],
    )
    n = merge_parallel_shards(tmp_path, n_workers=2)
    assert n == 3
    text = (tmp_path / "completions.jsonl").read_text()
    assert "e0" in text and "e1" in text and "e2" in text
    assert (tmp_path / "captures" / "index.jsonl").exists()


def test_nvidia_ready_configs_enable_parallel():
    basic = load_config(CONFIGS / "nvidia-qwen35-9b-500-basic.yaml")
    jlens = load_config(CONFIGS / "nvidia-qwen35-9b-500-jlens.yaml")
    assert basic.runtime.gpus == [0, 1]
    assert basic.runtime.parallel is True
    assert jlens.runtime.gpus == [0, 1]
    assert jlens.runtime.parallel is True
    assert basic.capture.enabled is False
    assert jlens.capture.enabled is True
    assert jlens.capture.top_k_tokens == 10


def test_cli_gpus_flag_parses():
    from gsm8k_jspace.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["run", "--config", "configs/smoke.yaml", "--gpus", "0,3"]
    )
    assert args.gpus == "0,3"
