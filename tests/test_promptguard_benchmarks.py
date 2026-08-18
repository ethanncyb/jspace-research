"""Offline tests for Olares routing, benchmark data, attacks, and metrics."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest
import torch

from promptguard import olares_client
from promptguard.baseline_attacks import ReNeLLMAttack, TAPAttack
from promptguard.benchmark_setup import (
    BenchmarkExample,
    BenchmarkSplit,
    heldout_split,
    load_advbench,
    load_harmbench,
    three_way_split,
)
from promptguard.config import (
    OlaresConfig,
    load_config,
    validate_frozen_guard10_defense,
)
from promptguard.drift_probe import DriftProbe
from promptguard.evolving_attacker import DetectionOutcome, StrategyPool
from promptguard.metrics import AttackTrial, summarize_trials
from promptguard.semantic_preservation import IntentPreservationJudge


def test_config_contains_poc_benchmark_and_olares_sections():
    config = load_config("config.yaml")
    assert 8 <= config.evolution.rounds <= 10
    assert config.data.use_benchmarks_for_evolution
    assert config.benchmarks.harmbench_subset_size == 25
    assert config.benchmarks.advbench_subset_size == 25
    assert config.olares.fast_model
    assert config.attacker.attempts_per_prompt == 24
    assert config.attacker.max_reflections_per_round == 12
    assert config.attacker.sandbox_min_score_reduction == pytest.approx(0.05)
    assert config.baselines.enabled == ["renellm", "tap", "random_transform"]
    assert config.baselines.query_budget == 48
    assert config.baselines.tap_candidates_per_iteration == 8
    assert config.baselines.tap_beam_width == 4
    validate_frozen_guard10_defense(config)


def test_frozen_guard10_rejects_weakened_defense_configuration():
    config = load_config("config.yaml")
    config.probe.threshold = 0.49
    with pytest.raises(ValueError, match="probe.threshold"):
        validate_frozen_guard10_defense(config)


def test_olares_midrun_failure_falls_back_per_call(monkeypatch):
    olares_client.configure(OlaresConfig())
    monkeypatch.setattr(olares_client, "USE_OLARES", True)
    monkeypatch.setattr(olares_client, "_ACTIVE_CHAT_API", "openai")

    def fail(*_args, **_kwargs):
        raise RuntimeError("network dropped")

    monkeypatch.setattr(olares_client, "_native_generate", fail)
    monkeypatch.setattr(
        olares_client,
        "_local_generate",
        lambda prompt, system, max_tokens: f"fallback:{prompt}",
    )
    assert olares_client.generate("hello") == "fallback:hello"


def test_olares_fast_model_uses_native_non_thinking_chat(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "candidate"}, "done_reason": "stop"}

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    config = OlaresConfig(timeout_seconds=12.0)
    olares_client.configure(config)
    monkeypatch.setattr(olares_client, "USE_OLARES", True)
    monkeypatch.setattr(olares_client.requests, "post", post)

    assert olares_client.generate("rewrite", model=config.fast_model) == "candidate"
    url, kwargs = calls[0]
    assert url.endswith("/api/chat")
    assert kwargs["json"]["stream"] is False
    assert kwargs["json"]["think"] is False
    assert kwargs["json"]["options"]["num_predict"] == 512
    assert kwargs["timeout"] == 12.0


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


def test_three_way_split_keeps_final_ids_fresh_and_disjoint():
    development = [
        BenchmarkExample("train-a", "harmbench", "a", "A"),
        BenchmarkExample("dev-b", "harmbench", "b", "B"),
        BenchmarkExample("train-c", "advbench", "c", "C"),
    ]
    universe = [
        *development,
        BenchmarkExample("fresh-1", "harmbench", "d", "D"),
        BenchmarkExample("fresh-2", "advbench", "e", "E"),
    ]
    split = three_way_split(
        development,
        universe,
        attacker_development_categories=["b"],
        final_test_size=2,
        final_test_seed=11,
    )

    training_ids = {example.id for example in split.training}
    development_ids = {example.id for example in split.attacker_development}
    final_ids = {example.id for example in split.final_test}
    assert development_ids == {"dev-b"}
    assert final_ids == {"fresh-1", "fresh-2"}
    assert not (training_ids & development_ids)
    assert not (training_ids & final_ids)
    assert not (development_ids & final_ids)


def test_intent_judge_is_structured_cached_and_fail_closed(tmp_path, monkeypatch):
    responses = iter(
        [
            '[{"index": 0, "preserves_intent": true, "confidence": 0.91}]',
            "not-json",
        ]
    )
    calls = []

    def generate(*_args, **_kwargs):
        calls.append(1)
        return next(responses)

    monkeypatch.setattr(olares_client, "generate", generate)
    judge = IntentPreservationJudge(
        model="judge",
        confidence_threshold=0.7,
        cache_path=tmp_path / "semantic.jsonl",
    )
    assert judge("request", "equivalent wrapper")
    assert judge("request", "equivalent wrapper")
    assert not judge("request", "changed intent")
    assert len(calls) == 2


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


def test_metrics_report_success_only_query_count_and_semantic_validity():
    rows = summarize_trials(
        [
            AttackTrial(
                "1", "harmbench", "test", 2, 5, True, 0.1,
                semantic_valid_attempts=2,
            ),
            AttackTrial(
                "2", "harmbench", "test", 5, 5, False, 0.9,
                semantic_valid_attempts=3,
            ),
        ]
    )
    dataset = next(row for row in rows if row.source_dataset == "harmbench")
    assert dataset.asr == pytest.approx(0.5)
    assert dataset.avg_queries_per_successful_evasion == pytest.approx(2.0)
    assert dataset.semantic_validity_rate == pytest.approx(5 / 7)


def test_benchmark_orchestration_writes_all_tables(tmp_path, monkeypatch):
    import promptguard.benchmark_runner as runner

    config = load_config("config.yaml")
    config.baselines.output_dir = str(tmp_path / "out")
    config.evolution.output_dir = str(tmp_path / "evolution")
    config.baselines.enabled = ["renellm", "tap", "random_transform"]
    config.baselines.query_budget = 2
    config.baselines.tap_candidates_per_iteration = 1
    config.baselines.tap_beam_width = 1
    config.baselines.tap_iterations = 2
    config.metrics.enable_harmfulness_judge = False
    config.semantic.cache_path = str(tmp_path / "semantic-cache.jsonl")
    examples = [
        BenchmarkExample("h1", "harmbench", "a", "first"),
        BenchmarkExample("a1", "advbench", "b", "second"),
    ]
    monkeypatch.setattr(
        runner, "load_model", lambda _config: SimpleNamespace(hooks=object())
    )
    monkeypatch.setattr(
        runner,
        "load_benchmark_splits",
        lambda _config: BenchmarkSplit((), (), tuple(examples)),
    )
    def generate(prompt, *_args, **_kwargs):
        if "preserves_intent" in prompt:
            items = json.loads(prompt.split("ITEMS_JSON: ", 1)[1])
            return json.dumps(
                [
                    {
                        "index": item["index"],
                        "preserves_intent": True,
                        "confidence": 0.99,
                    }
                    for item in items
                ]
            )
        return "candidate"

    monkeypatch.setattr(olares_client, "generate", generate)

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
    comparison_path = tmp_path / "out" / "final_comparison.csv"
    assert comparison_path.exists()
    assert (tmp_path / "out" / "attempts.csv").exists()
    assert (tmp_path / "out" / "cross_stage.csv").exists()
    with comparison_path.open(newline="", encoding="utf-8") as handle:
        comparison = list(csv.DictReader(handle))
    assert {row["method"] for row in comparison} == {
        "renellm",
        "tap",
        "random_transform",
        "evolving_attacker",
    }
    assert {int(row["query_budget"]) for row in comparison} == {2}
    assert all("query-budget-dependent" in row["budget_caveat"] for row in comparison)


def test_final_comparison_requires_completed_evolving_attacker_snapshot(
    tmp_path, monkeypatch
):
    import promptguard.benchmark_runner as runner

    config = load_config("config.yaml")
    config.evolution.output_dir = str(tmp_path / "unfinished-evolution")
    config.semantic.cache_path = str(tmp_path / "semantic.jsonl")
    monkeypatch.setattr(
        runner, "load_model", lambda _config: SimpleNamespace(hooks=object())
    )
    monkeypatch.setattr(
        runner,
        "load_benchmark_splits",
        lambda _config: BenchmarkSplit(
            (),
            (),
            (BenchmarkExample("fresh", "harmbench", "a", "final"),),
        ),
    )

    with pytest.raises(FileNotFoundError, match="finish the frozen"):
        runner.run_benchmarks(
            config,
            [],
            include_cross_stage=False,
            require_final_attacker_snapshot=True,
        )
