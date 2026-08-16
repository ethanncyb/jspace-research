"""Conservative positive-only continual updates for detector probe heads."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from probe import LayerProbeDetector


@dataclass
class FlaggedDeltaRecord:
    sample_id: str
    prompt: str
    score: float
    layer: int
    detector_version: str
    delta: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def fingerprint(self) -> str:
        stable = json.dumps(
            {
                "sample_id": self.sample_id,
                "prompt": self.prompt,
                "layer": self.layer,
                "detector_version": self.detector_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(stable.encode()).hexdigest()

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["delta"] = self.delta.detach().float().cpu().tolist()
        value["fingerprint"] = self.fingerprint
        return value

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> FlaggedDeltaRecord:
        fields = dict(value)
        fields.pop("fingerprint", None)
        fields["delta"] = torch.tensor(fields["delta"], dtype=torch.float32)
        return cls(**fields)


class ContinualUpdateStore:
    """Deduplicated auto-positive JSONL and lower-confidence review CSV."""

    REVIEW_FIELDS = (
        "fingerprint",
        "sample_id",
        "prompt",
        "score",
        "layer",
        "detector_version",
        "timestamp",
        "delta",
        "metadata",
    )

    def __init__(
        self,
        auto_buffer: str | Path,
        manual_review: str | Path,
        *,
        strict_confidence: float = 0.995,
    ) -> None:
        if not 0 <= strict_confidence <= 1:
            raise ValueError("strict confidence must be between 0 and 1")
        self.auto_buffer = Path(auto_buffer)
        self.manual_review = Path(manual_review)
        self.strict_confidence = strict_confidence
        self._fingerprints = self._load_fingerprints()

    def _load_fingerprints(self) -> set[str]:
        fingerprints: set[str] = set()
        if self.auto_buffer.exists():
            with self.auto_buffer.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        value = json.loads(line)
                        fingerprints.add(value.get("fingerprint", ""))
        if self.manual_review.exists():
            with self.manual_review.open(newline="", encoding="utf-8") as stream:
                fingerprints.update(
                    row["fingerprint"] for row in csv.DictReader(stream)
                )
        fingerprints.discard("")
        return fingerprints

    def route(self, record: FlaggedDeltaRecord) -> str:
        """Route a flagged prediction as auto-positive or manual review.

        This API intentionally has no path that creates an automatic benign
        label. Duplicate records are ignored across both destinations.
        """

        fingerprint = record.fingerprint
        if fingerprint in self._fingerprints:
            return "duplicate"
        self._fingerprints.add(fingerprint)
        if record.score >= self.strict_confidence:
            self.auto_buffer.parent.mkdir(parents=True, exist_ok=True)
            with self.auto_buffer.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record.to_json()) + "\n")
            return "auto_positive"

        self.manual_review.parent.mkdir(parents=True, exist_ok=True)
        exists = self.manual_review.exists() and self.manual_review.stat().st_size > 0
        with self.manual_review.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.REVIEW_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "fingerprint": fingerprint,
                    "sample_id": record.sample_id,
                    "prompt": record.prompt,
                    "score": record.score,
                    "layer": record.layer,
                    "detector_version": record.detector_version,
                    "timestamp": record.timestamp,
                    "delta": json.dumps(record.delta.detach().float().cpu().tolist()),
                    "metadata": json.dumps(record.metadata, sort_keys=True),
                }
            )
        return "manual_review"

    def auto_records(self) -> list[FlaggedDeltaRecord]:
        if not self.auto_buffer.exists():
            return []
        with self.auto_buffer.open(encoding="utf-8") as stream:
            return [
                FlaggedDeltaRecord.from_json(json.loads(line))
                for line in stream
                if line.strip()
            ]


@dataclass(frozen=True)
class FeatureExample:
    id: str
    label: int
    deltas: Mapping[int, torch.Tensor]


def balanced_trusted_replay(
    examples: Sequence[FeatureExample], *, seed: int = 0
) -> list[FeatureExample]:
    by_label = {0: [], 1: []}
    for example in examples:
        if example.label not in by_label:
            raise ValueError("trusted labels must be 0 or 1")
        by_label[example.label].append(example)
    count = min(map(len, by_label.values()))
    if count == 0:
        raise ValueError("trusted replay requires both benign and injected examples")
    rng = random.Random(seed)
    for values in by_label.values():
        rng.shuffle(values)
    replay = by_label[0][:count] + by_label[1][:count]
    rng.shuffle(replay)
    return replay


def cap_pseudo_labeled(
    trusted: Sequence[FeatureExample],
    pseudo: Sequence[FeatureExample],
    *,
    max_fraction: float,
    seed: int = 0,
) -> list[FeatureExample]:
    if not 0 <= max_fraction < 1:
        raise ValueError("max_fraction must be in [0, 1)")
    if max_fraction == 0 or not pseudo:
        return []
    maximum = int(len(trusted) * max_fraction / (1 - max_fraction))
    rng = random.Random(seed)
    selected = list(pseudo)
    rng.shuffle(selected)
    return selected[:maximum]


def estimate_diagonal_fisher(
    detector: LayerProbeDetector, examples: Sequence[FeatureExample]
) -> dict[str, torch.Tensor]:
    """Mean squared BCE gradients for trainable probe-head parameters."""

    fisher = {
        name: torch.zeros_like(param) for name, param in detector.named_parameters()
    }
    if not examples:
        return fisher
    detector.zero_grad(set_to_none=True)
    for example in examples:
        losses = []
        label = torch.tensor(float(example.label))
        for layer in detector.layers:
            losses.append(
                nn.functional.binary_cross_entropy_with_logits(
                    detector.logits(layer, example.deltas[layer]).reshape(()), label
                )
            )
        detector.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        for name, param in detector.named_parameters():
            if param.grad is not None:
                fisher[name].add_(param.grad.detach().square())
    for value in fisher.values():
        value.div_(len(examples))
    detector.zero_grad(set_to_none=True)
    return fisher


def ewc_penalty(
    detector: LayerProbeDetector,
    fisher: Mapping[str, torch.Tensor],
    anchor: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    penalties = [
        fisher[name].to(param.device) * (param - anchor[name].to(param.device)).square()
        for name, param in detector.named_parameters()
    ]
    if not penalties:
        return torch.tensor(0.0)
    return sum(value.sum() for value in penalties)


@dataclass(frozen=True)
class UpdateResult:
    published: bool
    rollback_path: Path
    metrics: Mapping[str, Any]
    trusted_count: int
    pseudo_count: int


def fine_tune_and_publish(
    detector: LayerProbeDetector,
    trusted_examples: Sequence[FeatureExample],
    pseudo_examples: Sequence[FeatureExample],
    destination: str | Path,
    *,
    safeguard: Callable[[LayerProbeDetector], tuple[bool, Mapping[str, Any]]],
    pseudo_fraction: float = 0.25,
    ewc_lambda: float = 1.0,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> UpdateResult:
    """Fine-tune heads, evaluate safeguards, and publish with rollback."""

    torch.manual_seed(seed)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rollback = destination.with_suffix(destination.suffix + ".rollback")
    if destination.exists():
        shutil.copy2(destination, rollback)
    else:
        detector.save(rollback)

    trusted = balanced_trusted_replay(trusted_examples, seed=seed)
    if any(example.label != 1 for example in pseudo_examples):
        raise ValueError("automatic continual examples must be positive labels only")
    pseudo = cap_pseudo_labeled(
        trusted, pseudo_examples, max_fraction=pseudo_fraction, seed=seed
    )
    fisher = estimate_diagonal_fisher(detector, trusted)
    anchor = {
        name: param.detach().clone() for name, param in detector.named_parameters()
    }
    optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        trusted_losses = []
        for example in trusted:
            label = torch.tensor(float(example.label))
            trusted_losses.extend(
                nn.functional.binary_cross_entropy_with_logits(
                    detector.logits(layer, example.deltas[layer]).reshape(()), label
                )
                for layer in detector.layers
            )
        pseudo_losses = []
        for example in pseudo:
            pseudo_losses.extend(
                nn.functional.binary_cross_entropy_with_logits(
                    detector.logits(layer, example.deltas[layer]).reshape(()),
                    torch.tensor(1.0),
                )
                for layer in detector.layers
            )
        loss = torch.stack(trusted_losses).mean()
        if pseudo_losses:
            loss = loss + torch.stack(pseudo_losses).mean()
        loss = loss + ewc_lambda * ewc_penalty(detector, fisher, anchor)
        loss.backward()
        optimizer.step()

    passed, metrics = safeguard(detector)
    if not passed:
        detector.load_state_dict(torch.load(rollback, weights_only=False)["state_dict"])
        return UpdateResult(False, rollback, metrics, len(trusted), len(pseudo))

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        detector.save(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return UpdateResult(True, rollback, metrics, len(trusted), len(pseudo))
