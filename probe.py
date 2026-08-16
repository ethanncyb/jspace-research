"""Dataset, linear probes, checkpointing, and dependency-free metrics."""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from model_hooks import validate_candidate_prefix


@dataclass(frozen=True)
class DatasetRow:
    id: str
    clean_prompt: str
    appended_text: str
    label: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pair_id(self) -> str:
        return str(self.metadata.get("pair_id", self.id))


DEFAULT_FIELD_MAP = {
    "id": "id",
    "clean_prompt": "clean_prompt",
    "appended_text": "appended_text",
    "candidate_prompt": "candidate_prompt",
    "label": "label",
}


def parse_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"0", "benign"}:
        return 0
    if normalized in {"1", "injected"}:
        return 1
    raise ValueError(f"unsupported label {value!r}; expected 0/1 or benign/injected")


def _raw_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise ValueError("JSON dataset must be an array")
    return value


def load_dataset(
    path: str | Path,
    *,
    field_map: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> list[DatasetRow]:
    """Load JSON, JSONL, or CSV rows into the canonical dataset contract."""

    mapping = {**DEFAULT_FIELD_MAP, **(field_map or {})}
    rows: list[DatasetRow] = []
    known_source_fields = set(mapping.values())
    for index, raw in enumerate(_raw_rows(Path(path))):
        candidate_suffix_ids: list[int] | None = None
        try:
            clean = str(raw[mapping["clean_prompt"]])
            label = parse_label(raw[mapping["label"]])
        except KeyError as exc:
            raise ValueError(
                f"row {index} is missing required field {exc.args[0]!r}"
            ) from exc
        row_id = str(raw.get(mapping["id"], index))
        appended_value = raw.get(mapping["appended_text"])
        candidate = raw.get(mapping["candidate_prompt"])
        if appended_value not in (None, ""):
            appended = str(appended_value)
        elif candidate not in (None, ""):
            if tokenizer is None:
                raise ValueError(
                    f"row {row_id}: tokenizer is required to derive candidate_prompt suffix"
                )
            suffix = validate_candidate_prefix(tokenizer, clean, str(candidate))
            candidate_suffix_ids = suffix[0].tolist()
            appended = tokenizer.decode(suffix[0].tolist(), skip_special_tokens=False)
        else:
            raise ValueError(
                f"row {row_id}: appended_text or candidate_prompt is required"
            )
        metadata = {
            key: value for key, value in raw.items() if key not in known_source_fields
        }
        if candidate_suffix_ids is not None:
            metadata["_candidate_suffix_token_ids"] = candidate_suffix_ids
        rows.append(DatasetRow(row_id, clean, appended, label, metadata))
    return rows


def stratified_group_split(
    rows: Sequence[DatasetRow], validation_fraction: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Seeded label-stratified split that never separates a pair ID."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row.pair_id, []).append(index)
    strata: dict[int, list[str]] = {0: [], 1: []}
    for group, indices in groups.items():
        labels = {rows[i].label for i in indices}
        # Mixed benign/injected pairs stay together and join the positive stratum.
        strata[max(labels)].append(group)
    rng = random.Random(seed)
    validation_groups: set[str] = set()
    for group_ids in strata.values():
        rng.shuffle(group_ids)
        if len(group_ids) <= 1:
            count = 0
        else:
            count = max(
                1, min(len(group_ids) - 1, round(len(group_ids) * validation_fraction))
            )
        validation_groups.update(group_ids[:count])
    train = [i for i, row in enumerate(rows) if row.pair_id not in validation_groups]
    validation = [i for i, row in enumerate(rows) if row.pair_id in validation_groups]
    if not train or not validation:
        raise ValueError("dataset has too few groups for a non-empty grouped split")
    return train, validation


def _key(layer: int) -> str:
    return f"layer_{layer}"


class LayerProbeDetector(nn.Module):
    """One normalized logistic head and benign reference per checkpoint."""

    def __init__(
        self,
        layers: Iterable[int],
        hidden_dim: int,
        *,
        model_id: str = "",
        aggregation: str = "max",
    ) -> None:
        super().__init__()
        self.layers = tuple(sorted(set(int(layer) for layer in layers)))
        self.hidden_dim = int(hidden_dim)
        self.model_id = model_id
        if aggregation not in {"max", "mean"}:
            raise ValueError("aggregation must be 'max' or 'mean'")
        self.aggregation = aggregation
        self.heads = nn.ModuleDict(
            {_key(layer): nn.Linear(self.hidden_dim, 1) for layer in self.layers}
        )
        for layer in self.layers:
            self.register_buffer(f"mean_{layer}", torch.zeros(self.hidden_dim))
            self.register_buffer(f"std_{layer}", torch.ones(self.hidden_dim))
            self.register_buffer(f"benign_{layer}", torch.zeros(self.hidden_dim))
            self.register_buffer(f"threshold_{layer}", torch.tensor(0.5))

    def _check(self, layer: int, delta: torch.Tensor) -> torch.Tensor:
        if layer not in self.layers:
            raise ValueError(f"layer {layer} is not present in this probe")
        if delta.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"probe hidden size {self.hidden_dim} is incompatible with delta size {delta.shape[-1]}"
            )
        return delta.float()

    def normalized(self, layer: int, delta: torch.Tensor) -> torch.Tensor:
        value = self._check(layer, delta)
        return (value - getattr(self, f"mean_{layer}")) / getattr(self, f"std_{layer}")

    def logits(self, layer: int, delta: torch.Tensor) -> torch.Tensor:
        return self.heads[_key(layer)](self.normalized(layer, delta)).squeeze(-1)

    def probability(self, layer: int, delta: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(layer, delta))

    def benign_distance(self, layer: int, delta: torch.Tensor) -> torch.Tensor:
        value = self._check(layer, delta)
        reference = getattr(self, f"benign_{layer}").to(value.device)
        return 1 - nn.functional.cosine_similarity(
            value.reshape(-1, self.hidden_dim),
            reference.expand(value.reshape(-1, self.hidden_dim).shape[0], -1),
            dim=-1,
            eps=1e-8,
        ).reshape(value.shape[:-1])

    def raw_attack_direction(self, layer: int) -> torch.Tensor:
        """Probe normal in raw activation coordinates (weight / feature std)."""

        weight = self.heads[_key(layer)].weight.squeeze(0)
        return weight / getattr(self, f"std_{layer}").clamp_min(1e-8)

    def threshold(self, layer: int) -> float:
        return float(getattr(self, f"threshold_{layer}").item())

    def set_statistics(
        self,
        layer: int,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        benign_reference: torch.Tensor,
        threshold: float | None = None,
    ) -> None:
        for name, value in {
            "mean": mean,
            "std": std.clamp_min(1e-6),
            "benign": benign_reference,
        }.items():
            target = getattr(self, f"{name}_{layer}")
            if value.shape != target.shape:
                raise ValueError(
                    f"{name} statistic has shape {value.shape}, expected {target.shape}"
                )
            target.copy_(value.detach().float())
        if threshold is not None:
            getattr(self, f"threshold_{layer}").fill_(float(threshold))

    def aggregate(self, values: Mapping[int, float | torch.Tensor]) -> float:
        if set(values) != set(self.layers):
            raise ValueError(
                "aggregate values must contain exactly the detector layers"
            )
        numbers = [
            float(torch.as_tensor(values[layer]).item()) for layer in self.layers
        ]
        return (
            max(numbers) if self.aggregation == "max" else sum(numbers) / len(numbers)
        )

    def predict(self, deltas: Mapping[int, torch.Tensor]) -> dict[str, Any]:
        probabilities = {
            layer: float(self.probability(layer, deltas[layer]).reshape(-1)[0].item())
            for layer in self.layers
        }
        distances = {
            layer: float(
                self.benign_distance(layer, deltas[layer]).reshape(-1)[0].item()
            )
            for layer in self.layers
        }
        score = self.aggregate(probabilities)
        return {
            "probabilities": probabilities,
            "benign_distances": distances,
            "score": score,
            "prediction": int(score >= 0.5),
        }

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "layers": self.layers,
            "hidden_dim": self.hidden_dim,
            "model_id": self.model_id,
            "aggregation": self.aggregation,
            "state_dict": self.state_dict(),
            "raw_attack_directions": {
                layer: self.raw_attack_direction(layer).detach().cpu()
                for layer in self.layers
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_layers: Iterable[int] | None = None,
        expected_hidden_dim: int | None = None,
        expected_model_id: str | None = None,
        map_location: str | torch.device = "cpu",
    ) -> LayerProbeDetector:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        layers = tuple(payload["layers"])
        hidden_dim = int(payload["hidden_dim"])
        if expected_layers is not None and tuple(expected_layers) != layers:
            raise ValueError(
                f"checkpoint layers {layers} do not match expected {tuple(expected_layers)}"
            )
        if expected_hidden_dim is not None and expected_hidden_dim != hidden_dim:
            raise ValueError(
                f"checkpoint hidden size {hidden_dim} does not match expected {expected_hidden_dim}"
            )
        if (
            expected_model_id is not None
            and payload.get("model_id") != expected_model_id
        ):
            raise ValueError(
                "checkpoint model ID is incompatible with the requested model"
            )
        detector = cls(
            layers,
            hidden_dim,
            model_id=payload.get("model_id", ""),
            aggregation=payload.get("aggregation", "max"),
        )
        detector.load_state_dict(payload["state_dict"])
        return detector


@dataclass(frozen=True)
class TrainingResult:
    detector: LayerProbeDetector
    train_indices: list[int]
    validation_indices: list[int]
    validation_metrics: dict[str, Any]


def train_probes(
    features: Mapping[int, torch.Tensor],
    labels: Sequence[int] | torch.Tensor,
    rows: Sequence[DatasetRow],
    *,
    model_id: str = "",
    aggregation: str = "max",
    validation_fraction: float = 0.2,
    epochs: int = 100,
    learning_rate: float = 1e-2,
    seed: int = 0,
) -> TrainingResult:
    """Fit all heads with a deterministic full-batch BCE objective."""

    torch.manual_seed(seed)
    labels_tensor = torch.as_tensor(labels, dtype=torch.float32)
    if len(rows) != len(labels_tensor):
        raise ValueError("rows and labels must have equal length")
    layers = tuple(sorted(features))
    if not layers:
        raise ValueError("at least one layer of features is required")
    hidden_dim = int(features[layers[0]].shape[-1])
    for layer, values in features.items():
        if values.shape != (len(rows), hidden_dim):
            raise ValueError(
                f"layer {layer} features have incompatible shape {tuple(values.shape)}"
            )
    train_indices, validation_indices = stratified_group_split(
        rows, validation_fraction, seed
    )
    detector = LayerProbeDetector(
        layers, hidden_dim, model_id=model_id, aggregation=aggregation
    )
    train_ix = torch.tensor(train_indices)
    for layer in layers:
        values = features[layer].detach().float()
        train_values = values[train_ix]
        mean = train_values.mean(dim=0)
        std = train_values.std(dim=0, unbiased=False).clamp_min(1e-6)
        benign_mask = labels_tensor[train_ix] == 0
        benign = (
            train_values[benign_mask].mean(dim=0)
            if benign_mask.any()
            else torch.zeros(hidden_dim)
        )
        detector.set_statistics(layer, mean=mean, std=std, benign_reference=benign)

    optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        losses = [
            nn.functional.binary_cross_entropy_with_logits(
                detector.logits(layer, features[layer][train_ix]),
                labels_tensor[train_ix],
            )
            for layer in layers
        ]
        torch.stack(losses).mean().backward()
        optimizer.step()

    probabilities = []
    with torch.no_grad():
        for index in validation_indices:
            per_layer = {
                layer: detector.probability(layer, features[layer][index]).item()
                for layer in layers
            }
            probabilities.append(detector.aggregate(per_layer))
    metrics = classification_metrics(
        labels_tensor[validation_indices].int().tolist(), probabilities
    )
    return TrainingResult(detector, train_indices, validation_indices, metrics)


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    # Mann-Whitney formulation with average ranks for ties.
    ordered = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def classification_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> dict[str, Any]:
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and equal length")
    predictions = [int(score >= threshold) for score in scores]
    pairs = list(zip(labels, predictions, strict=True))
    tp = sum(y == 1 and p == 1 for y, p in pairs)
    tn = sum(y == 0 and p == 0 for y, p in pairs)
    fp = sum(y == 0 and p == 1 for y, p in pairs)
    fn = sum(y == 1 and p == 0 for y, p in pairs)
    return {
        "accuracy": (tp + tn) / len(labels),
        "auroc": _auroc(labels, scores),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "false_positive_rate": fp / (fp + tn) if fp + tn else math.nan,
        "false_negative_rate": fn / (fn + tp) if fn + tp else math.nan,
        "count": len(labels),
    }


def evaluate(
    detector: LayerProbeDetector,
    features: Mapping[int, torch.Tensor],
    labels: Sequence[int],
) -> dict[str, Any]:
    scores = []
    with torch.no_grad():
        for index in range(len(labels)):
            scores.append(
                detector.aggregate(
                    {
                        layer: detector.probability(layer, features[layer][index])
                        for layer in detector.layers
                    }
                )
            )
    return classification_metrics(labels, scores)
