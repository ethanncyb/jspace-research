"""Benchmark baseline and evolving attacks against one or more drift probes."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from promptguard import olares_client
from promptguard.baseline_attacks import (
    AttackMethod,
    EvolvedPoolAttack,
    RandomTransformationAttack,
    ReNeLLMAttack,
    TAPAttack,
)
from promptguard.benchmark_setup import BenchmarkExample, load_benchmark_splits
from promptguard.config import (
    ResearchConfig,
    load_config,
    validate_frozen_guard10_defense,
)
from promptguard.drift_probe import DriftProbe
from promptguard.evolution_loop import CachedDeltaExtractor, make_evaluator
from promptguard.evolving_attacker import StrategyPool
from promptguard.metrics import (
    AttackTrial,
    harmfulness_score,
    summarize_trials,
)
from promptguard.model_hooks import load_model
from promptguard.semantic_preservation import IntentPreservationJudge

LOGGER = logging.getLogger(__name__)


def _round_number(path: str | Path) -> int:
    matches = re.findall(r"(\d+)", Path(path).stem)
    return int(matches[-1]) if matches else 0


def _static_methods(config: ResearchConfig, semantic_validator) -> list[AttackMethod]:
    methods: list[AttackMethod] = []
    for name in config.baselines.enabled:
        if name == "renellm":
            methods.append(
                ReNeLLMAttack(
                    model=config.olares.fast_model,
                    query_budget=config.baselines.query_budget,
                    semantic_validator=semantic_validator,
                )
            )
        elif name == "tap":
            methods.append(
                TAPAttack(
                    model=config.olares.fast_model,
                    query_budget=config.baselines.query_budget,
                    candidates_per_iteration=config.baselines.tap_candidates_per_iteration,
                    beam_width=config.baselines.tap_beam_width,
                    iterations=config.baselines.tap_iterations,
                    semantic_validator=semantic_validator,
                )
            )
        elif name == "random_transform":
            methods.append(
                RandomTransformationAttack(
                    query_budget=config.baselines.query_budget,
                    semantic_validator=semantic_validator,
                    seed=config.evolution.seed,
                )
            )
        else:
            raise ValueError(f"unknown baseline attack: {name}")
    return methods


def _strategy_snapshots(config: ResearchConfig) -> dict[int, StrategyPool]:
    output_dir = Path(config.evolution.output_dir)
    paths = sorted(output_dir.glob("strategy_pool_round_*.json"))
    snapshots = {0: StrategyPool()}
    snapshots.update({_round_number(path): StrategyPool.load(path) for path in paths})
    return snapshots


def _attempt_count(run) -> int:
    return run.attempts_used


def _run_method(
    method: AttackMethod,
    examples: Sequence[BenchmarkExample],
    evaluator,
    config: ResearchConfig,
    *,
    split_ids: set[str],
    enable_judge: bool | None = None,
) -> tuple[list[AttackTrial], list[dict[str, object]]]:
    trials = []
    details = []
    for example in examples:
        run = method.run(example.instruction, evaluator)
        best = run.best
        judge_score = None
        judge_enabled = (
            config.metrics.enable_harmfulness_judge
            if enable_judge is None
            else enable_judge
        )
        if judge_enabled:
            judge_score = harmfulness_score(
                example.instruction,
                best.prompt,
                model=config.olares.reason_model,
                max_tokens=config.metrics.judge_max_tokens,
            )
        trial = AttackTrial(
            instruction_id=example.id,
            source_dataset=example.source_dataset,
            method=method.name,
            attempts=_attempt_count(run),
            query_budget=run.query_budget,
            success=run.success,
            final_drift_score=best.drift_score,
            harmfulness_score=judge_score,
            semantic_valid_attempts=sum(
                attempt.semantic_valid for attempt in run.attempts
            ),
        )
        trials.append(trial)
        details.append(
            {
                **asdict(trial),
                "category": example.category,
                "split": "final_test" if example.id in split_ids else "unexpected",
                "best_prompt": best.prompt,
            }
        )
    return trials, details


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmarks(
    config: ResearchConfig,
    checkpoints: Sequence[str | Path],
    *,
    include_cross_stage: bool = True,
    require_final_attacker_snapshot: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    loaded = load_model(config.model)
    extractor = CachedDeltaExtractor(loaded.hooks, pooling=config.probe.pooling)
    split = load_benchmark_splits(config.benchmarks)
    final_examples = list(split.final_test)
    final_ids = {example.id for example in final_examples}
    if not config.semantic.enabled:
        raise ValueError("final comparison requires the semantic-preservation gate")
    semantic_validator = IntentPreservationJudge(
        model=config.semantic.model,
        use_native_model=config.semantic.use_native_model,
        model_think=config.semantic.model_think,
        confidence_threshold=config.semantic.confidence_threshold,
        max_tokens=config.semantic.judge_max_tokens,
        batch_size=config.semantic.batch_size,
        cache_path=config.semantic.cache_path,
    )
    snapshots = _strategy_snapshots(config)
    if (
        require_final_attacker_snapshot
        and config.evolution.rounds not in snapshots
    ):
        expected = (
            Path(config.evolution.output_dir)
            / f"strategy_pool_round_{config.evolution.rounds:03d}.json"
        )
        raise FileNotFoundError(
            f"final evolving-attacker snapshot is missing: {expected}; "
            "finish the frozen attacker-development run first"
        )
    metric_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    backend = olares_client.backend_name()
    output_dir = Path(config.baselines.output_dir)
    _write_csv(
        output_dir / "final_test_split.csv",
        [
            {
                "id": example.id,
                "source_dataset": example.source_dataset,
                "semantic_category": example.category,
                "split": "final_test",
                "instruction": example.instruction,
            }
            for example in final_examples
        ],
    )

    for checkpoint in checkpoints:
        guard_round = _round_number(checkpoint)
        probe = DriftProbe.load(checkpoint)
        evaluator = make_evaluator(
            probe,
            extractor,
            baseline=config.evolution.baseline_prompt,
            threshold=config.probe.threshold,
        )
        latest_snapshot_round = max(
            (round_index for round_index in snapshots if round_index <= guard_round),
            default=min(snapshots),
        )
        methods = [
            *_static_methods(config, semantic_validator),
            EvolvedPoolAttack(
                snapshots[latest_snapshot_round],
                query_budget=config.baselines.query_budget,
                semantic_validator=semantic_validator,
            ),
        ]
        for method in methods:
            trials, details = _run_method(
                method,
                final_examples,
                evaluator,
                config,
                split_ids=final_ids,
            )
            detail_rows.extend(
                {
                    "guard_round": guard_round,
                    "strategy_round": latest_snapshot_round,
                    "backend": backend,
                    **row,
                }
                for row in details
            )
            pool_size = len(snapshots[latest_snapshot_round])
            metric_rows.extend(
                {
                    "guard_round": guard_round,
                    "strategy_round": latest_snapshot_round,
                    "strategy_pool_size": pool_size,
                    "backend": backend,
                    **asdict(row),
                }
                for row in summarize_trials(trials)
            )

        if not include_cross_stage:
            continue
        # DARWIN Figure-4-style Attack-K x Guard-J matrix. This is intentionally
        # probe-only, so it does not spend additional Olares generation calls.
        for strategy_round, pool in sorted(snapshots.items()):
            method = EvolvedPoolAttack(
                pool,
                query_budget=config.baselines.query_budget,
                semantic_validator=semantic_validator,
            )
            trials, _ = _run_method(
                method,
                final_examples,
                evaluator,
                config,
                split_ids=final_ids,
                enable_judge=False,
            )
            for row in summarize_trials(trials):
                cross_rows.append(
                    {
                        "attack_round": strategy_round,
                        "guard_round": guard_round,
                        "strategy_pool_size": len(pool),
                        "backend": backend,
                        **asdict(row),
                    }
                )

    _write_csv(output_dir / "asr_aqc.csv", metric_rows)
    _write_csv(
        output_dir / "final_comparison.csv",
        [row for row in metric_rows if row["source_dataset"] == "all"],
    )
    _write_csv(output_dir / "attempts.csv", detail_rows)
    _write_csv(output_dir / "cross_stage.csv", cross_rows)
    return metric_rows, cross_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="probe checkpoints; defaults to all outputs/checkpoints/guard_*.pt",
    )
    parser.add_argument(
        "--frozen-guard-checkpoint",
        help="evaluate every attack method against exactly this immutable guard checkpoint",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    if args.frozen_guard_checkpoint:
        validate_frozen_guard10_defense(config)
    olares_client.configure(config.olares)
    olares_client.check_olares_reachable()
    LOGGER.info("active inference backend: %s", olares_client.backend_name())
    if args.checkpoints and args.frozen_guard_checkpoint:
        raise ValueError(
            "use either --checkpoints or --frozen-guard-checkpoint, not both"
        )
    checkpoints = (
        [args.frozen_guard_checkpoint]
        if args.frozen_guard_checkpoint
        else args.checkpoints
        or sorted(Path(config.evolution.checkpoint_dir).glob("guard_*.pt"))
    )
    if not checkpoints:
        raise FileNotFoundError("no probe checkpoints found; run evolution_loop first")
    metrics, cross = run_benchmarks(
        config,
        checkpoints,
        include_cross_stage=not bool(args.frozen_guard_checkpoint),
        require_final_attacker_snapshot=bool(args.frozen_guard_checkpoint),
    )
    print(
        "CAUTION: ASR is query-budget-dependent; the final comparison is valid "
        "only because every method uses the same configured detector-query budget."
    )
    print(f"Wrote {len(metrics)} metric rows and {len(cross)} cross-stage rows")


if __name__ == "__main__":
    main()
