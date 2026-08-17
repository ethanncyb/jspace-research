"""Offline tests for Olares routing, benchmark data, attacks, and metrics."""

from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest
import torch

from promptguard import olares_client
from promptguard.baseline_attacks import ReNeLLMAttack, TAPAttack
from promptguard.benchmark_setup import (
    BenchmarkExample,
    heldout_split,
    load_advbench,
    load_harmbench,
)
from promptguard.config import OlaresConfig, load_config
from promptguard.drift_probe import DriftProbe
from promptguard.evolving_attacker import DetectionOutcome, StrategyPool
from promptguard.metrics import AttackTrial, summarize_trials


def test_config_contains_poc_benchmark_and_olares_sections():
    config = load_config("config.yaml")
    assert 8 <= config.evolution.rounds <= 10
    assert config.data.use_benchmarks_for_evolution
    assert config.benchmarks.harmbench_subset_size == 25
    assert config.benchmarks.advbench_subset_size == 25
    assert config.olares.fast_model
    assert config.baselines.enabled == ["renellm"]


def test_olares_midrun_failure_falls_back_per_call(monkeypatch):
    olares_client.configure(OlaresConfig())
    monkeypatch.setattr(olares_client, "USE_OLARES", True)
    monkeypatch.setattr(olares_client, "_ACTIVE_CHAT_API", "openai")

    def fail(*_args, **_kwargs):
        raise RuntimeError("network dropped")

    monkeypatch.setattr(olares_client, "_openai_generate", fail)
    monkeypatch.setattr(
        olares_client,
        "_local_generate",
        lambda prompt, system, max_tokens: f"fallback:{prompt}",
    )
    assert olares_client.generate("hello") == "fallback:hello"


def test_olares_health_check_sets_active_backend(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "OK"}}]}

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(olares_client.requests, "post", post)
    olares_client.configure(OlaresConfig(timeout_seconds=0.01))
    assert olares_client.check_olares_reachable()
    assert olares_client.USE_OLARES
    assert olares_client.backend_name() == "olares:openai"
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url.endswith("/v1/chat/completions")
    assert kwargs["headers"]["Authorization"].startswith("Bearer ")
    assert kwargs["json"]["messages"] == [
        {"role": "user", "content": "Reply with OK."}
    ]
    assert kwargs["timeout"] == 10.0


def test_olares_embedding_has_noncrashing_hash_fallback(monkeypatch):
    olares_client.configure(OlaresConfig(force_local_fallback=True))

    def fail():
        raise RuntimeError("no local embedding model")

    monkeypatch.setattr(olares_client, "_get_local_embedder", fail)
    first = olares_client.embed("same text")
    second = olares_client.embed("same text")
    assert first == second
    assert len(first) == 384


def test_benchmark_loaders_normalize_and_hold_out_categories(tmp_path):
    harmbench = tmp_path / "harmbench.csv"
    with harmbench.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["BehaviorID", "SemanticCategory", "Behavior"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"BehaviorID": "h1", "SemanticCategory": "cat-a", "Behavior": "A"},
                {"BehaviorID": "h2", "SemanticCategory": "cat-b", "Behavior": "B"},
                {"BehaviorID": "h3", "SemanticCategory": "cat-a", "Behavior": "C"},
            ]
        )
    advbench = tmp_path / "advbench.csv"
    with advbench.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["goal", "target"])
        writer.writeheader()
        writer.writerow({"goal": "D", "target": "target"})

    harm = load_harmbench(harmbench, cache_dir=tmp_path, subset_size="full", seed=1)
    adv = load_advbench(advbench, cache_dir=tmp_path, subset_size=1, seed=1)
    assert {example.source_dataset for example in [*harm, *adv]} == {
        "harmbench",
        "advbench",
    }
    split = heldout_split(harm, heldout_categories=["cat-b"], seed=1)
    assert {example.category for example in split.heldout} == {"cat-b"}
    assert all(example.category != "cat-b" for example in split.train)


def test_strategy_pool_snapshot_roundtrip(tmp_path):
    pool = StrategyPool()
    destination = tmp_path / "pool.json"
    pool.save(destination)
    restored = StrategyPool.load(destination)
    assert [item.name for item in restored.strategies] == [
        item.name for item in pool.strategies
    ]


def test_baselines_use_shared_probe_evaluator(monkeypatch):
    monkeypatch.setattr(
        olares_client, "generate", lambda *_args, **_kwargs: "candidate"
    )

    def evaluator(prompt):
        return DetectionOutcome(
            0.1 if prompt == "candidate" else 0.9, prompt != "candidate"
        )

    renellm = ReNeLLMAttack(model="fast")
    assert renellm.run("instruction", evaluator).success
    tap = TAPAttack(
        model="fast",
        query_budget=2,
        candidates_per_iteration=1,
        beam_width=1,
        iterations=1,
    )
    assert tap.run("instruction", evaluator).success


def test_metrics_report_asr_and_budget_exhaustion_aqc():
    rows = summarize_trials(
        [
            AttackTrial("1", "harmbench", "test", 2, 5, True, 0.1),
            AttackTrial("2", "harmbench", "test", 5, 5, False, 0.9),
        ]
    )
    dataset = next(row for row in rows if row.source_dataset == "harmbench")
    assert dataset.asr == pytest.approx(0.5)
    assert dataset.aqc == pytest.approx(3.5)


def test_benchmark_orchestration_writes_all_tables(tmp_path, monkeypatch):
    import promptguard.benchmark_runner as runner

    config = load_config("config.yaml")
    config.baselines.output_dir = str(tmp_path / "out")
    config.evolution.output_dir = str(tmp_path / "evolution")
    config.baselines.enabled = ["renellm"]
    config.metrics.enable_harmfulness_judge = False
    examples = [
        BenchmarkExample("h1", "harmbench", "a", "first"),
        BenchmarkExample("a1", "advbench", "b", "second"),
    ]
    monkeypatch.setattr(
        runner, "load_model", lambda _config: SimpleNamespace(hooks=object())
    )
    monkeypatch.setattr(runner, "load_benchmarks", lambda _config: examples)
    monkeypatch.setattr(
        olares_client, "generate", lambda *_args, **_kwargs: "candidate"
    )

    class Extractor:
        def __init__(self, _model, *, pooling):
            pass

        def __call__(self, _baseline, text):
            value = -1.0 if "candidate" in text else 1.0
            return {0: torch.tensor([[value, 0.0]])}

    monkeypatch.setattr(runner, "CachedDeltaExtractor", Extractor)
    probe = DriftProbe([0], 2)
    with torch.no_grad():
        probe.classifiers["0"].weight[:] = torch.tensor([[1.0, 0.0]])
        probe.classifiers["0"].bias.zero_()
    checkpoint = tmp_path / "guard_000.pt"
    probe.save(checkpoint)

    metrics, cross = runner.run_benchmarks(config, [checkpoint])
    assert metrics and cross
    assert (tmp_path / "out" / "asr_aqc.csv").exists()
    assert (tmp_path / "out" / "attempts.csv").exists()
    assert (tmp_path / "out" / "cross_stage.csv").exists()
