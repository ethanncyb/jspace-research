"""Continual DARWIN-Guard-style updates for the activation drift probe."""

from __future__ import annotations

import csv
import itertools
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from promptguard.config import GuardConfig, ProbeConfig
from promptguard.drift_probe import DeltaExample, DriftProbe
from promptguard.evolving_attacker import AttackRecord, StrategyPool

LOGGER = logging.getLogger(__name__)

DeltaExtractor = Callable[[str, str], DeltaExample]


@dataclass(frozen=True)
class SelfLabeledExample:
    text: str
    predicted_label: int
    confidence: float
    source: str = "self_labeled"
    strategy_name: str | None = None


@dataclass(frozen=True)
class LabeledPrompt:
    baseline: str
    text: str
    label: int
    group: str
    source: str


@dataclass
class GuardRoundResult:
    round_index: int
    checkpoint_path: Path
    accepted_count: int
    review_count: int
    training_count: int
    final_loss: float
    candidate_count: int = 0
    discarded_count: int = 0


class ConfidenceGate:
    """Admit high-confidence pseudo-labels and queue the rest for review."""

    def __init__(self, threshold: float, review_queue_path: str | Path) -> None:
        if not 0.5 <= threshold <= 1:
            raise ValueError("confidence gate threshold must be in [0.5,1]")
        self.threshold = threshold
        self.review_queue_path = Path(review_queue_path)

    def partition(
        self, examples: Sequence[SelfLabeledExample], *, round_index: int
    ) -> tuple[list[SelfLabeledExample], list[SelfLabeledExample]]:
        accepted = [
            example for example in examples if example.confidence >= self.threshold
        ]
        review = [
            example for example in examples if example.confidence < self.threshold
        ]
        if review:
            self._append_review(review, round_index)
        return accepted, review

    def _append_review(
        self, examples: Sequence[SelfLabeledExample], round_index: int
    ) -> None:
        self.review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        append = self.review_queue_path.exists() and round_index != 1
        with self.review_queue_path.open(
            "a" if append else "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["round", "text", "predicted_label", "confidence", "source"],
            )
            if not append:
                writer.writeheader()
            for example in examples:
                writer.writerow(
                    {
                        "round": round_index,
                        "text": example.text,
                        "predicted_label": example.predicted_label,
                        "confidence": example.confidence,
                        "source": example.source,
                    }
                )


class EvolvingGuard:
    """Retrain the existing probe in-place on current and replayed examples."""

    def __init__(
        self,
        probe: DriftProbe,
        *,
        strategies: StrategyPool,
        probe_config: ProbeConfig,
        guard_config: GuardConfig,
        seed: int = 7,
    ) -> None:
        self.probe = probe
        self.strategies = strategies
        self.probe_config = probe_config
        self.guard_config = guard_config
        self.seed = seed
        self.gate = ConfidenceGate(
            guard_config.confidence_gate, guard_config.review_queue_path
        )
        self._replay: list[LabeledPrompt] = []

    def _candidate_examples(
        self, attack_records: Sequence[AttackRecord]
    ) -> tuple[list[SelfLabeledExample], int]:
        """Create conservative pseudo-label candidates from non-escaping attempts.

        Successful attacks already have a trusted unsafe label and must not be
        self-labeled as benign. Duplicate or malformed records are discarded.
        """

        candidates: list[SelfLabeledExample] = []
        discarded = 0
        seen: set[str] = set()
        for record in attack_records:
            if record.success:
                continue
            if (
                not record.transformed_prompt.strip()
                or record.transformed_prompt in seen
                or not math.isfinite(record.score)
                or not 0.0 <= record.score <= 1.0
            ):
                discarded += 1
                continue
            seen.add(record.transformed_prompt)
            candidates.append(
                SelfLabeledExample(
                    text=record.transformed_prompt,
                    predicted_label=1 if record.flagged else 0,
                    confidence=record.score if record.flagged else 1.0 - record.score,
                    source="online_attack_attempt",
                    strategy_name=record.strategy_name,
                )
            )
        return candidates, discarded

    def _matched_examples(
        self,
        successful_attacks: Sequence[AttackRecord],
        benign_prompts: Sequence[str],
        baseline: str,
    ) -> list[LabeledPrompt]:
        if not benign_prompts:
            raise ValueError("at least one benign prompt is required")
        result: list[LabeledPrompt] = []
        benign_cycle = itertools.cycle(benign_prompts)
        for attack in successful_attacks:
            benign = next(benign_cycle)
            strategy = self.strategies.by_name(attack.strategy_name)
            result.extend(
                [
                    LabeledPrompt(
                        baseline,
                        attack.transformed_prompt,
                        1,
                        "disguised",
                        "trusted_attack_seed",
                    ),
                    LabeledPrompt(
                        baseline,
                        strategy.transform(benign),
                        0,
                        "disguised",
                        "matched_benign",
                    ),
                    LabeledPrompt(
                        baseline, attack.payload, 1, "raw", "trusted_attack_seed"
                    ),
                    LabeledPrompt(baseline, benign, 0, "raw", "matched_benign"),
                ]
            )
        return result

    def run_round(
        self,
        *,
        round_index: int,
        successful_attacks: Sequence[AttackRecord],
        benign_prompts: Sequence[str],
        baseline: str,
        extractor: DeltaExtractor,
        checkpoint_path: str | Path,
        self_labeled: Sequence[SelfLabeledExample] = (),
        attack_records: Sequence[AttackRecord] = (),
    ) -> GuardRoundResult:
        """Continue training from ``self.probe``; never reinitialize weights."""

        generated, discarded = self._candidate_examples(attack_records)
        candidates = [*generated, *self_labeled]
        accepted, review = self.gate.partition(candidates, round_index=round_index)
        LOGGER.info(
            "guard_candidate_summary round=%d generated=%d external=%d total=%d confidence_gate=%.6f passed=%d queued_for_review=%d discarded=%d",
            round_index,
            len(generated),
            len(self_labeled),
            len(candidates),
            self.gate.threshold,
            len(accepted),
            len(review),
            discarded,
        )
        if not candidates:
            LOGGER.warning(
                "guard_candidate_trace round=%d reason=no_non_escaping_attack_attempts_and_no_external_candidates attack_records=%d successful_records=%d",
                round_index,
                len(attack_records),
                sum(record.success for record in attack_records),
            )
        current = self._matched_examples(successful_attacks, benign_prompts, baseline)
        current.extend(
            LabeledPrompt(
                baseline,
                example.text,
                example.predicted_label,
                "disguised",
                example.source,
            )
            for example in accepted
        )
        benign_cycle = itertools.cycle(benign_prompts)
        strategy_cycle = itertools.cycle(self.strategies.strategies)
        for example in accepted:
            benign = next(benign_cycle)
            try:
                strategy = (
                    self.strategies.by_name(example.strategy_name)
                    if example.strategy_name
                    else next(strategy_cycle)
                )
            except KeyError:
                strategy = next(strategy_cycle)
            current.append(
                LabeledPrompt(
                    baseline,
                    strategy.transform(benign),
                    0,
                    "disguised",
                    "pseudo_matched_benign",
                )
            )
        LOGGER.info(
            "guard_training_balance round=%d trusted_attack_pairs=%d pseudo_unsafe=%d pseudo_matched_benign=%d",
            round_index,
            len(successful_attacks),
            len(accepted),
            len(accepted),
        )
        if not current:
            # A perfectly blocking guard yields no successful attacks. Save an
            # unchanged checkpoint so cross-stage evaluation remains complete.
            destination = Path(checkpoint_path)
            self.probe.save(destination, round_index=round_index, unchanged=True)
            return GuardRoundResult(
                round_index,
                destination,
                len(accepted),
                len(review),
                0,
                0.0,
                len(candidates),
                discarded,
            )

        training = list(current)
        if self.guard_config.replay_previous_rounds:
            training.extend(self._replay)
        examples = [extractor(item.baseline, item.text) for item in training]
        labels = [item.label for item in training]
        weights = [
            self.guard_config.lambda_raw if item.group == "raw" else 1.0
            for item in training
        ]
        losses = self.probe.fit(
            examples,
            labels,
            epochs=self.probe_config.epochs,
            learning_rate=self.probe_config.learning_rate,
            batch_size=self.probe_config.batch_size,
            weight_decay=self.probe_config.weight_decay,
            layer_loss_weight=self.probe_config.layer_loss_weight,
            sample_weights=weights,
            seed=self.seed + round_index,
        )
        self._replay.extend(current)
        destination = Path(checkpoint_path)
        self.probe.save(
            destination,
            round_index=round_index,
            lambda_raw=self.guard_config.lambda_raw,
        )
        return GuardRoundResult(
            round_index=round_index,
            checkpoint_path=destination,
            accepted_count=len(accepted),
            review_count=len(review),
            training_count=len(training),
            final_loss=losses[-1],
            candidate_count=len(candidates),
            discarded_count=discarded,
        )
