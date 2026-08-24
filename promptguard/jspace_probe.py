"""Injection probe over pooled JSpace concept features.

Reuses :class:`~promptguard.drift_probe.DriftProbe`'s training, scoring, and
evaluation loop verbatim; only the per-layer head and the checkpoint format
differ. Inputs are featurized traces ``{layer: Tensor[n_concepts]}`` produced
by :func:`promptguard.jspace_features.featurize` (batch dim optional, same
convention as ``DeltaExample``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from promptguard.drift_probe import DriftProbe

HEAD_MODES = ("linear", "mlp")


class JSpaceProbe(DriftProbe):
    """Per-layer probe heads over pooled JSpace concept features.

    ``head="linear"`` is a logistic probe (``nn.Linear(n_concepts, 1)`` per
    layer, aggregated as in ``DriftProbe``). ``head="mlp"`` uses a small
    per-layer MLP (``n_concepts -> mlp_hidden -> 1`` with GELU + dropout),
    still aggregated over layers by mean/max logits.
    """

    def __init__(
        self,
        layer_indices: Iterable[int],
        n_concepts: int,
        *,
        aggregate: str = "mean_logits",
        head: str = "linear",
        mlp_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        if head not in HEAD_MODES:
            raise ValueError(f"unknown head mode: {head}")
        if n_concepts <= 0:
            raise ValueError("n_concepts must be positive")
        self.n_concepts = n_concepts
        self.head = head
        self.mlp_hidden = mlp_hidden
        self.dropout = dropout
        super().__init__(layer_indices, n_concepts, aggregate=aggregate)
        if head == "mlp":
            self.classifiers = nn.ModuleDict(
                {
                    str(layer): nn.Sequential(
                        nn.Linear(n_concepts, mlp_hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(mlp_hidden, 1),
                    )
                    for layer in self.layer_indices
                }
            )

    def feature_direction(self, layer: int) -> torch.Tensor:
        """Unit probe-weight vector over concept features (linear head only)."""

        if self.head != "linear":
            raise ValueError(
                "feature_direction is only defined for the linear head"
            )
        return self.drift_direction(layer)

    def save(self, path: str | Path, **metadata: Any) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "layer_indices": self.layer_indices,
                "n_concepts": self.n_concepts,
                "aggregate": self.aggregate,
                "head": self.head,
                "mlp_hidden": self.mlp_hidden,
                "dropout": self.dropout,
                "metadata": metadata,
            },
            destination,
        )

    @classmethod
    def load(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> JSpaceProbe:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        probe = cls(
            payload["layer_indices"],
            payload["n_concepts"],
            aggregate=payload.get("aggregate", "mean_logits"),
            head=payload.get("head", "linear"),
            mlp_hidden=payload.get("mlp_hidden", 64),
            dropout=payload.get("dropout", 0.1),
        )
        probe.load_state_dict(payload["state_dict"])
        probe.eval()
        return probe
