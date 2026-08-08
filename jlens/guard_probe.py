# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Inference-time security probe with an experience memory of jailbreaks.

Inspired by DARWIN-Guard but post-training: no weight updates, learning is
appending to a persistent :class:`~jlens.guard_memory.GuardMemory`. Only
incidents that actually bypassed the model (``succeeded=True``) can ever
cause a block.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from jlens.guard_embed import embed_text, normalize_prompt
from jlens.guard_memory import GuardMemory, Incident


@dataclass
class Verdict:
    """Outcome of assessing one prompt.

    Attributes:
        action: One of ``"allow"``, ``"flag"``, or ``"block"``.
        score: Best cosine similarity found against successful incidents.
        nearest: The corresponding incident; ``None`` for a clean allow.
        reason: Human-readable explanation of the decision.
    """

    action: str
    score: float
    nearest: Incident | None
    reason: str


class SecurityProbe:
    """Guard that blocks prompts resembling previously successful jailbreaks.

    A prompt is blocked when its nearest successful incident from the same
    user beats ``repeat_offender_threshold`` (this user has bypassed before)
    or when the nearest successful incident from any user beats
    ``block_threshold``. Scores above ``flag_threshold`` but below the
    blocking bar produce a flag; anything lower is allowed.
    """

    def __init__(
        self,
        memory: GuardMemory,
        block_threshold: float = 0.85,
        repeat_offender_threshold: float = 0.75,
        flag_threshold: float = 0.60,
    ) -> None:
        self.memory = memory
        self.block_threshold = block_threshold
        self.repeat_offender_threshold = repeat_offender_threshold
        self.flag_threshold = flag_threshold

    def assess(self, prompt: str, user_id: str) -> Verdict:
        """Decide whether to allow, flag, or block ``prompt`` for ``user_id``."""
        embedding = embed_text(prompt)
        any_incident, any_score = self._nearest(
            self.memory.similar(embedding, top_k=1, succeeded=True)
        )
        user_incident, user_score = self._nearest(
            self.memory.similar(embedding, top_k=1, user_id=user_id, succeeded=True)
        )
        best_score = max(any_score, user_score)

        if user_score >= self.repeat_offender_threshold:
            return Verdict(
                action="block",
                score=user_score,
                nearest=user_incident,
                reason=(
                    f"repeat offender: prompt is {user_score:.3f} similar to a "
                    "jailbreak this user previously succeeded with"
                ),
            )
        if any_score >= self.block_threshold:
            return Verdict(
                action="block",
                score=any_score,
                nearest=any_incident,
                reason=(
                    f"prompt is {any_score:.3f} similar to a known successful jailbreak"
                ),
            )
        if best_score >= self.flag_threshold:
            return Verdict(
                action="flag",
                score=best_score,
                nearest=any_incident if any_score >= user_score else user_incident,
                reason=(
                    f"prompt is {best_score:.3f} similar to a known successful "
                    "jailbreak, below the blocking threshold"
                ),
            )
        return Verdict(
            action="allow",
            score=best_score,
            nearest=None,
            reason="no sufficiently similar successful jailbreak on record",
        )

    def learn(
        self,
        prompt: str,
        user_id: str,
        succeeded: bool,
        strategy: str | None = None,
    ) -> Incident:
        """Record an attempt in memory and return the stored incident."""
        incident = Incident(
            prompt=prompt,
            normalized=normalize_prompt(prompt),
            embedding=embed_text(prompt).tolist(),
            user_id=user_id,
            strategy=strategy,
            succeeded=succeeded,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return self.memory.add(incident)

    @staticmethod
    def _nearest(
        hits: list[tuple[Incident, float]],
    ) -> tuple[Incident | None, float]:
        return hits[0] if hits else (None, 0.0)
