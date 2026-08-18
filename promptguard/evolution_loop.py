"""Round-based attacker/guard co-evolution and cross-stage evaluation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from promptguard import olares_client
from promptguard.benchmark_setup import BenchmarkSplit, heldout_split, load_benchmarks
from promptguard.config import (
    ResearchConfig,
    load_config,
    validate_frozen_guard10_defense,
)
from promptguard.drift_probe import DeltaExample, DriftProbe, extract_appended_delta
from promptguard.evolving_attacker import (
    AttackRecord,
    DetectionOutcome,
    EvolvingAttacker,
    OlaresReflector,
    PromptStrategy,
    heldout_attack_strategies,
    starter_strategies,
)
from promptguard.evolving_guard import EvolvingGuard
from promptguard.model_hooks import HookedModel, load_model
from promptguard.semantic_preservation import IntentPreservationJudge

LOGGER = logging.getLogger(__name__)


def seeded_prompt_order(prompts: Sequence[str], *, seed: int) -> list[str]:
    """Return a deterministic shuffled copy without mutating configuration data."""

    ordered = list(prompts)
    random.Random(seed).shuffle(ordered)
    return ordered


class CachedDeltaExtractor:
    """Avoid rerunning the 4B backbone for repeated prompt pairs."""

    def __init__(self, model: HookedModel, *, pooling: str) -> None:
        self.model = model
        self.pooling = pooling
        self.cache: dict[tuple[str, str], dict[int, torch.Tensor]] = {}

    def __call__(self, baseline: str, text: str) -> dict[int, torch.Tensor]:
        key = (baseline, text)
        if key not in self.cache:
            values = extract_appended_delta(
                self.model, baseline, text, pooling=self.pooling
            )
            self.cache[key] = {
                layer: value.detach().cpu() for layer, value in values.items()
            }
        return self.cache[key]


def make_evaluator(
    probe: DriftProbe,
    extractor: CachedDeltaExtractor,
    *,
    baseline: str,
    threshold: float,
):
    def evaluate(prompt: str) -> DetectionOutcome:
        score = float(probe.score(extractor(baseline, prompt))[0].reshape(-1)[0])
        return DetectionOutcome(score=score, flagged=score >= threshold)

    return evaluate


@dataclass
class CrossStageRow:
    attack_round: int | str
    guard_round: int
    attack_family: str
    asr: float
    count: int


@dataclass
class HeldoutFamilyRow:
    round: int
    attack_family: str
    asr: float
    count: int


def _append_dataclass_rows(
    rows: Sequence[HeldoutFamilyRow], csv_path: str | Path, *, round_index: int
) -> None:
    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    append = destination.exists() and round_index != 1
    with destination.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        if not append:
            writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def evaluation_attack_strategies() -> tuple[PromptStrategy, ...]:
    """Return exhaustive, reporting-friendly families for held-out payloads.

    Translation and acrostic/encoding remain evaluation-only transformations.
    The starter strategies are also evaluated on payloads that never enter guard
    training, which makes categories such as roleplay directly comparable.
    """

    reporting_families = {
        "heldout_translation": "translated",
        "heldout_acrostic": "acrostic_encoding",
        "heldout_indirection": "indirection",
        "heldout_fullwidth": "unicode_encoding",
        "roleplay": "roleplay",
    }
    strategies: list[PromptStrategy] = []
    for strategy in (*heldout_attack_strategies(), *starter_strategies()):
        family = reporting_families.get(strategy.name, strategy.family)
        strategies.append(
            PromptStrategy(
                strategy.name,
                family,
                strategy.transform,
                origin="heldout_evaluation",
                description=strategy.description,
                template=strategy.template,
            )
        )
    return tuple(strategies)


def active_family_evaluate(
    records: Sequence[AttackRecord],
    *,
    round_index: int,
    payload_count: int,
    csv_path: str | Path | None = None,
) -> list[HeldoutFamilyRow]:
    """Summarize sampled active attacks by family and unique payload."""

    attempted: dict[str, set[int]] = {}
    evaded: dict[str, set[int]] = {}
    all_evaded: set[int] = set()
    for record in records:
        attempted.setdefault(record.strategy_family, set()).add(record.payload_id)
        if record.success:
            evaded.setdefault(record.strategy_family, set()).add(record.payload_id)
            all_evaded.add(record.payload_id)
    rows = [
        HeldoutFamilyRow(
            round_index,
            family,
            len(evaded.get(family, set())) / len(payload_ids),
            len(payload_ids),
        )
        for family, payload_ids in sorted(attempted.items())
    ]
    rows.append(
        HeldoutFamilyRow(
            round_index,
            "all_active",
            len(all_evaded) / max(payload_count, 1),
            payload_count,
        )
    )
    if csv_path is not None:
        _append_dataclass_rows(rows, csv_path, round_index=round_index)
    return rows


def heldout_family_evaluate(
    probe: DriftProbe,
    family_examples: Mapping[str, Sequence[DeltaExample]],
    *,
    round_index: int,
    threshold: float,
    csv_path: str | Path | None = None,
) -> tuple[list[HeldoutFamilyRow], float]:
    """Evaluate never-trained attack families against the current guard."""

    rows: list[HeldoutFamilyRow] = []
    total_evaded = 0
    total_count = 0
    for family, examples in family_examples.items():
        evaded = sum(
            float(probe.score(example)[0].reshape(-1)[0]) < threshold
            for example in examples
        )
        rows.append(
            HeldoutFamilyRow(
                round=round_index,
                attack_family=family,
                asr=evaded / max(len(examples), 1),
                count=len(examples),
            )
        )
        total_evaded += evaded
        total_count += len(examples)
    overall = total_evaded / max(total_count, 1)
    rows.append(HeldoutFamilyRow(round_index, "all_heldout", overall, total_count))
    if csv_path is not None:
        _append_dataclass_rows(rows, csv_path, round_index=round_index)
    LOGGER.info(
        "heldout_round_summary round=%d overall_asr=%.6f families=%s",
        round_index,
        overall,
        {row.attack_family: row.asr for row in rows[:-1]},
    )
    return rows, overall


def cross_stage_evaluate(
    attack_sets: Mapping[int, Sequence[DeltaExample]],
    guard_checkpoints: Mapping[int, str | Path],
    *,
    threshold: float,
    heldout: Sequence[DeltaExample] = (),
    csv_path: str | Path | None = None,
) -> list[CrossStageRow]:
    """Evaluate every Attack-K against every saved Guard-J checkpoint."""

    rows: list[CrossStageRow] = []
    for guard_round, checkpoint in sorted(guard_checkpoints.items()):
        guard = DriftProbe.load(checkpoint)
        for attack_round, examples in sorted(attack_sets.items()):
            evaded = sum(
                float(guard.score(example)[0].reshape(-1)[0]) < threshold
                for example in examples
            )
            rows.append(
                CrossStageRow(
                    attack_round,
                    guard_round,
                    "evolved",
                    evaded / max(len(examples), 1),
                    len(examples),
                )
            )
        if heldout:
            evaded = sum(
                float(guard.score(example)[0].reshape(-1)[0]) < threshold
                for example in heldout
            )
            rows.append(
                CrossStageRow(
                    "heldout",
                    guard_round,
                    "heldout_untrained",
                    evaded / len(heldout),
                    len(heldout),
                )
            )
    if csv_path is not None:
        destination = Path(csv_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(asdict(rows[0]))
                if rows
                else ["attack_round", "guard_round", "attack_family", "asr", "count"],
            )
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    return rows


class EvolutionRunner:
    def __init__(
        self,
        config: ResearchConfig,
        model: HookedModel,
        *,
        probe: DriftProbe | None = None,
        frozen_guard_checkpoint: str | Path | None = None,
    ) -> None:
        self.config = config
        self.model = model
        # Seed before constructing the probe; seeding only in ``run`` leaves
        # its initial classifier weights nondeterministic across executions.
        random.seed(config.evolution.seed)
        torch.manual_seed(config.evolution.seed)
        self.extractor = CachedDeltaExtractor(model, pooling=config.probe.pooling)
        self.probe = probe or DriftProbe(
            model.full_attention_layers,
            model.hidden_dim,
            aggregate=config.probe.aggregate,
        )
        self._provided_probe = probe is not None
        self.frozen_guard_checkpoint = (
            Path(frozen_guard_checkpoint) if frozen_guard_checkpoint else None
        )
        if self.frozen_guard_checkpoint is not None and probe is None:
            raise ValueError("frozen_guard_checkpoint requires a loaded probe")
        self.malicious_prompts = list(config.data.malicious_prompts)
        self.heldout_prompts = list(config.data.heldout_malicious_prompts)
        self.benchmark_split: BenchmarkSplit | None = None
        if config.data.use_benchmarks_for_evolution:
            self.benchmark_split = heldout_split(
                load_benchmarks(config.benchmarks),
                heldout_categories=(
                    config.benchmarks.attacker_development_categories
                ),
                heldout_fraction=(
                    config.benchmarks.attacker_development_fraction
                ),
                seed=config.benchmarks.seed,
            )
            guard_training_prompts = [
                example.instruction for example in self.benchmark_split.training
            ]
            self.heldout_prompts = [
                example.instruction
                for example in self.benchmark_split.attacker_development
            ]
            self.malicious_prompts = (
                self.heldout_prompts
                if self.frozen_guard_checkpoint is not None
                else guard_training_prompts
            )
        self.malicious_prompts = seeded_prompt_order(
            self.malicious_prompts, seed=config.evolution.seed
        )
        semantic_validator = (
            IntentPreservationJudge(
                model=config.semantic.model,
                use_native_model=config.semantic.use_native_model,
                model_think=config.semantic.model_think,
                confidence_threshold=config.semantic.confidence_threshold,
                max_tokens=config.semantic.judge_max_tokens,
                batch_size=config.semantic.batch_size,
                cache_path=config.semantic.cache_path,
            )
            if config.semantic.enabled
            else None
        )
        self.attacker = EvolvingAttacker(
            alpha=config.attacker.alpha,
            gamma=config.attacker.gamma,
            sandbox_tau=config.attacker.sandbox_tau,
            sandbox_min_score_reduction=(
                config.attacker.sandbox_min_score_reduction
            ),
            attempts_per_prompt=config.attacker.attempts_per_prompt,
            reflector=OlaresReflector(config.olares.fast_model),
            embedder=lambda text: olares_client.embed(
                text, model=config.olares.embed_model
            ),
            dedup_similarity=config.attacker.dedup_similarity,
            max_reflections_per_round=config.attacker.max_reflections_per_round,
            threshold=config.probe.threshold,
            semantic_validator=semantic_validator,
            seed=config.evolution.seed,
        )
        self.guard = EvolvingGuard(
            self.probe,
            strategies=self.attacker.pool,
            probe_config=config.probe,
            guard_config=config.guard,
            seed=config.evolution.seed,
        )
        self.output_dir = Path(config.evolution.output_dir)
        self.checkpoint_dir = Path(config.evolution.checkpoint_dir)

    def _write_benchmark_manifest(self) -> None:
        if self.benchmark_split is None:
            return
        path = self.output_dir / "benchmark_split.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        development_ids = {
            example.id for example in self.benchmark_split.attacker_development
        }
        with path.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "id",
                "source_dataset",
                "semantic_category",
                "split",
                "instruction",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for example in (
                *self.benchmark_split.training,
                *self.benchmark_split.attacker_development,
            ):
                writer.writerow(
                    {
                        "id": example.id,
                        "source_dataset": example.source_dataset,
                        "semantic_category": example.category,
                        "split": (
                            "attacker_development"
                            if example.id in development_ids
                            else "guard_training"
                        ),
                        "instruction": example.instruction,
                    }
                )

    def _features(self, prompts: Sequence[str]) -> list[DeltaExample]:
        baseline = self.config.evolution.baseline_prompt
        return [self.extractor(baseline, prompt) for prompt in prompts]

    def bootstrap(self) -> Path:
        """Fit Guard-0 on raw trusted examples before adaptive red-teaming."""

        benign = self.config.data.benign_prompts
        malicious = self.malicious_prompts
        if not benign or not malicious:
            raise ValueError(
                "config.data needs non-empty benign_prompts and malicious_prompts"
            )
        if not self._provided_probe:
            examples = self._features([*benign, *malicious])
            labels = [0] * len(benign) + [1] * len(malicious)
            probe_config = self.config.probe
            self.probe.fit(
                examples,
                labels,
                epochs=probe_config.epochs,
                learning_rate=probe_config.learning_rate,
                batch_size=probe_config.batch_size,
                weight_decay=probe_config.weight_decay,
                layer_loss_weight=probe_config.layer_loss_weight,
                seed=self.config.evolution.seed,
            )
        checkpoint = self.checkpoint_dir / "guard_000.pt"
        stage = "continued_checkpoint" if self._provided_probe else "bootstrap"
        self.probe.save(checkpoint, round_index=0, stage=stage)
        return checkpoint

    def _append_round_metrics(self, values: dict[str, Any]) -> None:
        path = self.output_dir / "round_metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        append = path.exists() and values.get("round") != 1
        with path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values))
            if not append:
                writer.writeheader()
            writer.writerow(values)

    def _log_round_one_eval_examples(
        self, prompts: Sequence[str], labels: Sequence[int], *, threshold: float
    ) -> None:
        labeled = list(zip(prompts, labels, strict=True))
        sample = [item for item in labeled if item[1] == 0][:4] + [
            item for item in labeled if item[1] == 1
        ][:4]
        for prompt, label in sample:
            score = float(
                self.probe.score(
                    self.extractor(self.config.evolution.baseline_prompt, prompt)
                )[0].reshape(-1)[0]
            )
            LOGGER.info(
                "round1_eval_example true_label=%s prompt=%r drift_score=%.6f probe_prediction=%s",
                "unsafe" if label else "benign",
                prompt,
                score,
                "unsafe" if score >= threshold else "benign",
            )

    def run(self) -> list[dict[str, Any]]:
        random.seed(self.config.evolution.seed)
        torch.manual_seed(self.config.evolution.seed)
        self._write_benchmark_manifest()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "seed": self.config.evolution.seed,
                    "benchmark_seed": self.config.benchmarks.seed,
                    "active_prompt_count": len(self.malicious_prompts),
                    "heldout_prompt_count": len(self.heldout_prompts),
                    "rounds": self.config.evolution.rounds,
                    "guard_frozen": self.frozen_guard_checkpoint is not None,
                    "frozen_guard_checkpoint": (
                        str(self.frozen_guard_checkpoint)
                        if self.frozen_guard_checkpoint is not None
                        else None
                    ),
                    "detector_threshold": self.config.probe.threshold,
                    "intervention_mode": self.config.intervention.mode,
                    "intervention_threshold": self.config.intervention.threshold,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.attacker.pool.save(self.output_dir / "strategy_pool_round_000.json")
        if self.frozen_guard_checkpoint is None:
            guard_checkpoints: dict[int, Path] = {0: self.bootstrap()}
        else:
            matches = re.findall(r"(\d+)", self.frozen_guard_checkpoint.stem)
            frozen_round = int(matches[-1]) if matches else 0
            guard_checkpoints = {frozen_round: self.frozen_guard_checkpoint}
            LOGGER.info(
                "frozen_guard_enabled checkpoint=%s guard_round=%d threshold=%.6f intervention_mode=%s intervention_threshold=%.6f",
                self.frozen_guard_checkpoint,
                frozen_round,
                self.config.probe.threshold,
                self.config.intervention.mode,
                self.config.intervention.threshold,
            )
        attack_sets: dict[int, list[DeltaExample]] = {}
        heldout_payloads = self.heldout_prompts or self.malicious_prompts
        heldout_families = evaluation_attack_strategies()
        heldout_prompts_by_family: dict[str, list[str]] = {}
        for strategy in heldout_families:
            heldout_prompts_by_family.setdefault(strategy.family, []).extend(
                strategy.transform(prompt) for prompt in heldout_payloads
            )
        heldout_by_family = {
            family: self._features(prompts)
            for family, prompts in heldout_prompts_by_family.items()
        }
        heldout = [
            example
            for family_examples in heldout_by_family.values()
            for example in family_examples
        ]
        hard_eval_prompts = [
            strategy.transform(prompt)
            for strategy in heldout_families
            for prompt in self.config.data.benign_prompts
        ] + [
            strategy.transform(prompt)
            for strategy in heldout_families
            for prompt in heldout_payloads
        ]
        hard_eval_labels = [0] * (
            len(heldout_families) * len(self.config.data.benign_prompts)
        ) + [1] * (len(heldout_families) * len(heldout_payloads))
        summaries: list[dict[str, Any]] = []
        baseline = self.config.evolution.baseline_prompt
        threshold = self.config.probe.threshold
        validation_payloads = (self.heldout_prompts or self.malicious_prompts)[
            : self.config.attacker.mutation_validation_samples
        ]
        self.attacker.prewarm_semantic_cache(validation_payloads)

        for round_index in range(1, self.config.evolution.rounds + 1):
            if round_index == 1:
                self._log_round_one_eval_examples(
                    hard_eval_prompts, hard_eval_labels, threshold=threshold
                )
            evaluator = make_evaluator(
                self.probe,
                self.extractor,
                baseline=baseline,
                threshold=threshold,
            )
            # Mutation admission uses a frozen copy, so results do not shift if
            # guard training is later made asynchronous.
            validation_probe = copy.deepcopy(self.probe).cpu().eval()
            heldout_evaluator = make_evaluator(
                validation_probe,
                self.extractor,
                baseline=baseline,
                threshold=threshold,
            )
            attack_result = self.attacker.run_round(
                round_index=round_index,
                payloads=self.malicious_prompts,
                evaluator=evaluator,
                validation_payloads=validation_payloads,
                heldout_evaluator=heldout_evaluator,
                csv_path=self.output_dir / "attacks.csv",
            )
            active_family_evaluate(
                attack_result.records,
                round_index=round_index,
                payload_count=len(self.malicious_prompts),
                csv_path=self.output_dir / "active_by_round.csv",
            )
            unique_prompts = list(
                dict.fromkeys(
                    record.transformed_prompt for record in attack_result.records
                )
            )
            attack_sets[round_index] = self._features(unique_prompts)

            guard_result = None
            if self.frozen_guard_checkpoint is None:
                checkpoint = self.checkpoint_dir / f"guard_{round_index:03d}.pt"
                guard_result = self.guard.run_round(
                    round_index=round_index,
                    successful_attacks=attack_result.successful_attacks,
                    benign_prompts=self.config.data.benign_prompts,
                    baseline=baseline,
                    extractor=self.extractor,
                    checkpoint_path=checkpoint,
                    attack_records=attack_result.records,
                )
                guard_checkpoints[round_index] = checkpoint
            else:
                LOGGER.info(
                    "guard_update_skipped round=%d reason=frozen_guard checkpoint=%s",
                    round_index,
                    self.frozen_guard_checkpoint,
                )
            self.attacker.pool.save(
                self.output_dir / f"strategy_pool_round_{round_index:03d}.json"
            )
            metrics = self.probe.evaluate(
                self._features(hard_eval_prompts),
                hard_eval_labels,
                threshold=threshold,
            )
            heldout_rows, heldout_asr = heldout_family_evaluate(
                self.probe,
                heldout_by_family,
                round_index=round_index,
                threshold=threshold,
                csv_path=self.output_dir / "heldout_by_round.csv",
            )
            summary = {
                "round": round_index,
                "strategy_pool_size": len(self.attacker.pool),
                "attacker_asr": attack_result.asr,
                "heldout_attacker_asr": heldout_asr,
                "probe_unsafe_recall": metrics.unsafe_recall,
                "probe_benign_pass_rate": metrics.benign_pass_rate,
                "accepted_pseudo_labels": guard_result.accepted_count if guard_result else 0,
                "review_queue_count": guard_result.review_count if guard_result else 0,
                "pseudo_label_candidates": guard_result.candidate_count if guard_result else 0,
                "discarded_pseudo_labels": guard_result.discarded_count if guard_result else 0,
                "guard_update_enabled": guard_result is not None,
                "reflection_calls": attack_result.reflection_calls,
                "reflection_rejections": attack_result.reflection_rejections,
                "admitted_strategies": len(attack_result.admitted_strategies),
            }
            summary.update(
                {
                    f"heldout_asr_{row.attack_family}": row.asr
                    for row in heldout_rows
                    if row.attack_family != "all_heldout"
                }
            )
            summaries.append(summary)
            self._append_round_metrics(summary)
            cross_stage_evaluate(
                attack_sets,
                guard_checkpoints,
                threshold=threshold,
                heldout=heldout,
                csv_path=self.output_dir / "cross_stage.csv",
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "run_summary.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
        return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--probe", help="continue from an existing probe checkpoint instead of Guard-0"
    )
    parser.add_argument(
        "--frozen-guard-checkpoint",
        help=(
            "load an already-trained guard checkpoint and disable bootstrap and "
            "all guard updates for the full attacker run"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "override evolution.seed for probe initialization, strategy sampling, "
            "guard training, and active-prompt order"
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="override evolution.output_dir (useful for keeping seeded runs separate)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="override evolution.checkpoint_dir",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.frozen_guard_checkpoint:
        validate_frozen_guard10_defense(config)
    if args.seed is not None:
        config.evolution.seed = args.seed
    if args.output_dir:
        config.evolution.output_dir = args.output_dir
        config.guard.review_queue_path = str(Path(args.output_dir) / "review_queue.csv")
    if args.checkpoint_dir:
        config.evolution.checkpoint_dir = args.checkpoint_dir
    olares_client.configure(config.olares)
    olares_client.check_olares_reachable()
    print(f"Inference backend: {olares_client.backend_name()}")
    loaded = load_model(config.model)
    if args.probe and args.frozen_guard_checkpoint:
        raise ValueError("use either --probe or --frozen-guard-checkpoint, not both")
    probe_path = args.frozen_guard_checkpoint or args.probe
    probe = DriftProbe.load(probe_path) if probe_path else None
    summaries = EvolutionRunner(
        config,
        loaded.hooks,
        probe=probe,
        frozen_guard_checkpoint=args.frozen_guard_checkpoint,
    ).run()
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
