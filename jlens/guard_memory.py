# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Persistent incident memory for the security probe.

Incidents are kept in memory and appended to a JSONL file, so the probe's
experience survives restarts. Similarity search is brute-force cosine over
the stored embeddings — the memory is meant to be small.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from jlens.guard_embed import cosine


@dataclass
class Incident:
    """One recorded jailbreak attempt.

    Attributes:
        prompt: The raw prompt text.
        normalized: :func:`jlens.guard_embed.normalize_prompt` of the prompt.
        embedding: The prompt embedding as a plain list, or ``None``.
        user_id: Identifier of the user who issued the prompt.
        strategy: Jailbreak strategy identifier, if known.
        succeeded: Whether the attempt bypassed the model's refusals.
        timestamp: ISO 8601 UTC timestamp of when the incident was recorded.
    """

    prompt: str
    normalized: str
    embedding: list[float] | None
    user_id: str
    strategy: str | None
    succeeded: bool
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> dict:
        """Serialize to a plain dict (embedding stays a list of floats)."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> Incident:
        """Rebuild an incident from a dict parsed from one JSONL line."""
        return cls(**data)


class GuardMemory:
    """Append-only incident store backed by a JSONL file.

    The whole file is loaded on init; :meth:`add` appends one JSON line per
    incident. Parent directories of ``path`` are created on first write.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._incidents: list[Incident] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self._incidents.append(Incident.from_json(json.loads(line)))

    def __len__(self) -> int:
        return len(self._incidents)

    def add(self, incident: Incident) -> Incident:
        """Append ``incident`` to the store and persist it as one JSON line."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(incident.to_json()) + "\n")
        self._incidents.append(incident)
        return incident

    def query(
        self,
        user_id: str | None = None,
        strategy: str | None = None,
        succeeded: bool | None = None,
    ) -> list[Incident]:
        """All incidents matching the given filters (``None`` = no filter)."""
        return [
            inc
            for inc in self._incidents
            if (user_id is None or inc.user_id == user_id)
            and (strategy is None or inc.strategy == strategy)
            and (succeeded is None or inc.succeeded == succeeded)
        ]

    def similar(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        user_id: str | None = None,
        succeeded: bool | None = None,
    ) -> list[tuple[Incident, float]]:
        """Brute-force cosine search over stored embeddings, best first.

        Only incidents with a non-``None`` embedding are considered;
        ``user_id`` and ``succeeded`` filter the candidates. Returns
        ``(incident, score)`` pairs sorted by descending score; ties go
        to the most recent timestamp (ISO 8601 UTC strings sort
        chronologically).
        """
        scored = [
            (cosine(embedding, np.asarray(inc.embedding)), inc)
            for inc in self._incidents
            if inc.embedding is not None
            and (user_id is None or inc.user_id == user_id)
            and (succeeded is None or inc.succeeded == succeeded)
        ]
        scored.sort(key=lambda pair: (pair[0], pair[1].timestamp), reverse=True)
        return [(inc, score) for score, inc in scored[:top_k]]
