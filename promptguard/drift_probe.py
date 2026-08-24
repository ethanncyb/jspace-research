"""Activation-delta extraction and a layerwise linear drift probe."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

ActivationMap = Mapping[int, torch.Tensor]
DeltaExample = Mapping[int, torch.Tensor]


def pool_activations(
    activations: ActivationMap,
    attention_mask: torch.Tensor,
    *,
    pooling: str = "last_token",
) -> dict[int, torch.Tensor]:
    """Pool ``[batch, sequence, hidden]`` residuals to ``[batch, hidden]``."""

    mask = attention_mask.to(dtype=torch.bool)
    pooled: dict[int, torch.Tensor] = {}
    for layer, hidden in activations.items():
        if hidden.ndim != 3:
            raise ValueError(f"layer {layer}: expected [batch, seq, hidden]")
        layer_mask = mask.to(hidden.device)
        if pooling == "last_token":
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            last = (layer_mask.long() * positions.unsqueeze(0)).max(dim=1).values
            pooled[layer] = hidden[
                torch.arange(hidden.shape[0], device=hidden.device), last
            ]
        elif pooling == "mean":
            weights = layer_mask.unsqueeze(-1).to(hidden.dtype)
            pooled[layer] = (hidden * weights).sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1)
        else:
            raise ValueError(f"unknown pooling mode: {pooling}")
    return pooled


def compute_activation_deltas(
    before: ActivationMap,
    after: ActivationMap,
    before_mask: torch.Tensor,
    after_mask: torch.Tensor,
    *,
    pooling: str = "last_token",
) -> dict[int, torch.Tensor]:
    """Compute pooled ``after - before`` residual-stream changes per layer."""

    if set(before) != set(after):
        raise ValueError("before/after captures contain different layers")
    before_pooled = pool_activations(before, before_mask, pooling=pooling)
    after_pooled = pool_activations(after, after_mask, pooling=pooling)
    return {
        layer: after_pooled[layer].float() - before_pooled[layer].float()
        for layer in before_pooled
    }


def extract_delta(
    hooked_model: Any,
    before_text: str | list[str],
    after_text: str | list[str],
    *,
    pooling: str = "last_token",
) -> dict[int, torch.Tensor]:
    """Run paired capture passes and return activation deltas."""

    before, before_mask = hooked_model.capture(before_text)
    after, after_mask = hooked_model.capture(after_text)
    return compute_activation_deltas(
        before, after, before_mask, after_mask, pooling=pooling
    )


def extract_appended_delta(
    hooked_model: Any,
    base_text: str,
    appended_text: str,
    *,
    pooling: str = "last_token",
    separator: str = "\n",
) -> dict[int, torch.Tensor]:
    """Compare a clean prefix with the same prefix plus an injected segment."""

    combined = f"{base_text}{separator}{appended_text}"
    return extract_delta(hooked_model, base_text, combined, pooling=pooling)


@dataclass
class ProbeMetrics:
    accuracy: float
    auc: float
    false_positive_rate: float
    false_negative_rate: float
    unsafe_recall: float
    benign_pass_rate: float
    count: int


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """ROC AUC via pairwise ranking, including half credit for ties."""

    positives = [s for y, s in zip(labels, scores, strict=True) if y == 1]
    negatives = [s for y, s in zip(labels, scores, strict=True) if y == 0]
    if not positives or not negatives:
        return math.nan
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


class DriftProbe(nn.Module):
    """One logistic probe per layer, aggregated by the mean of their logits."""

    def __init__(
        self,
        layer_indices: Iterable[int],
        hidden_dim: int,
        *,
        aggregate: str = "mean_logits",
    ) -> None:
        super().__init__()
        self.layer_indices = sorted(set(layer_indices))
        if not self.layer_indices:
            raise ValueError("at least one probe layer is required")
        self.hidden_dim = hidden_dim
        self.aggregate = aggregate
        self.classifiers = nn.ModuleDict(
            {str(layer): nn.Linear(hidden_dim, 1) for layer in self.layer_indices}
        )

    def _validate(self, deltas: DeltaExample) -> None:
        missing = set(self.layer_indices) - set(deltas)
        if missing:
            raise ValueError(f"activation deltas missing layers: {sorted(missing)}")

    def layer_logits(self, deltas: DeltaExample) -> dict[int, torch.Tensor]:
        self._validate(deltas)
        result = {}
        for layer in self.layer_indices:
            value = deltas[layer].float()
            if value.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"layer {layer}: hidden dim {value.shape[-1]}, "
                    f"expected {self.hidden_dim}"
                )
            target = self.classifiers[str(layer)]
            value = value.to(next(target.parameters()).device)
            result[layer] = target(value).squeeze(-1)
        return result

    def _aggregate_logits(
        self, layer_logits: Mapping[int, torch.Tensor]
    ) -> torch.Tensor:
        logits = torch.stack(list(layer_logits.values()), dim=-1)
        if self.aggregate == "mean_logits":
            return logits.mean(dim=-1)
        if self.aggregate == "max_logits":
            return logits.max(dim=-1).values
        raise ValueError(f"unknown aggregate mode: {self.aggregate}")

    def forward(self, deltas: DeltaExample) -> torch.Tensor:
        return self._aggregate_logits(self.layer_logits(deltas))

    @torch.inference_mode()
    def score(
        self, deltas: DeltaExample
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        self.eval()
        layer_logits = self.layer_logits(deltas)
        aggregate = self._aggregate_logits(layer_logits)
        return torch.sigmoid(aggregate), {
            layer: torch.sigmoid(logit) for layer, logit in layer_logits.items()
        }

    @torch.inference_mode()
    def drift_direction(self, layer: int) -> torch.Tensor:
        """Return the unit probe-weight vector used for circuit breaking."""

        weight = self.classifiers[str(layer)].weight[0]
        return weight / weight.norm().clamp_min(1e-12)

    def fit(
        self,
        examples: Sequence[DeltaExample],
        labels: Sequence[int],
        *,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        weight_decay: float = 1e-4,
        layer_loss_weight: float = 0.25,
        sample_weights: Sequence[float] | None = None,
        seed: int = 7,
    ) -> list[float]:
        """Train in-place, preserving existing weights for continual updates."""

        if len(examples) != len(labels) or not examples:
            raise ValueError("examples and labels must be equally sized and non-empty")
        if sample_weights is not None and len(sample_weights) != len(examples):
            raise ValueError("sample_weights length must match examples")
        device = next(self.parameters()).device
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        generator = torch.Generator().manual_seed(seed)
        losses: list[float] = []
        self.train()
        for _epoch in range(epochs):
            permutation = torch.randperm(len(examples), generator=generator).tolist()
            epoch_loss = 0.0
            for start in range(0, len(examples), batch_size):
                ids = permutation[start : start + batch_size]
                batch = {
                    layer: torch.stack(
                        [examples[index][layer].reshape(-1) for index in ids]
                    ).to(device)
                    for layer in self.layer_indices
                }
                targets = torch.tensor(
                    [labels[index] for index in ids], dtype=torch.float32, device=device
                )
                weights = torch.ones_like(targets)
                if sample_weights is not None:
                    weights = torch.tensor(
                        [sample_weights[index] for index in ids],
                        dtype=torch.float32,
                        device=device,
                    )
                per_layer = self.layer_logits(batch)
                aggregate = self._aggregate_logits(per_layer)
                aggregate_loss = nn.functional.binary_cross_entropy_with_logits(
                    aggregate, targets, reduction="none"
                )
                layer_loss = torch.stack(
                    [
                        nn.functional.binary_cross_entropy_with_logits(
                            logits, targets, reduction="none"
                        )
                        for logits in per_layer.values()
                    ],
                    dim=-1,
                ).mean(dim=-1)
                loss = (
                    (aggregate_loss + layer_loss_weight * layer_loss) * weights
                ).sum() / weights.sum().clamp_min(1e-12)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(ids)
            losses.append(epoch_loss / len(examples))
        self.eval()
        return losses

    def evaluate(
        self,
        examples: Sequence[DeltaExample],
        labels: Sequence[int],
        *,
        threshold: float = 0.5,
    ) -> ProbeMetrics:
        scores = [float(self.score(example)[0].reshape(-1)[0]) for example in examples]
        predictions = [int(score >= threshold) for score in scores]
        tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels, strict=True))
        tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels, strict=True))
        fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels, strict=True))
        fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels, strict=True))
        return ProbeMetrics(
            accuracy=(tp + tn) / max(len(labels), 1),
            auc=binary_auc(labels, scores),
            false_positive_rate=fp / max(fp + tn, 1),
            false_negative_rate=fn / max(fn + tp, 1),
            unsafe_recall=tp / max(tp + fn, 1),
            benign_pass_rate=tn / max(tn + fp, 1),
            count=len(labels),
        )

    def save(self, path: str | Path, **metadata: Any) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "layer_indices": self.layer_indices,
                "hidden_dim": self.hidden_dim,
                "aggregate": self.aggregate,
                "metadata": metadata,
            },
            destination,
        )

    @classmethod
    def load(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> DriftProbe:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        probe = cls(
            payload["layer_indices"],
            payload["hidden_dim"],
            aggregate=payload.get("aggregate", "mean_logits"),
        )
        probe.load_state_dict(payload["state_dict"])
        probe.eval()
        return probe


def metrics_dict(metrics: ProbeMetrics) -> dict[str, float | int]:
    return asdict(metrics)
