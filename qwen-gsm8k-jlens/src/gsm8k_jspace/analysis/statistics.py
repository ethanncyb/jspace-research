"""Paired statistics helpers used by comparison and notebooks."""

from __future__ import annotations

from gsm8k_jspace.evaluation.compare import _bootstrap_diff, _mcnemar_exact


def mcnemar_p(broken: int, fixed: int) -> float:
    return _mcnemar_exact(broken, fixed)


def bootstrap_accuracy_diff(baseline: list[bool], candidate: list[bool], seed: int = 0):
    return _bootstrap_diff(baseline, candidate, seed=seed)
