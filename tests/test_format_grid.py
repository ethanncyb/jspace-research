from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from jspace_research.phase4.format_grid import (
    FORMATS,
    WRAPPERS,
    build_grid,
    check_identity,
    load_done,
    repair_tail,
    score_grid,
    summarize,
)


class FakeScorer:
    dictionary: torch.Tensor = torch.zeros(1)

    def score(self, residual: torch.Tensor, dictionary: torch.Tensor) -> dict[str, Any]:
        value: float = float(residual.sum())
        return {
            "mean_score": value,
            "mean_prediction": value > 0.5,
            "logistic_score": value,
            "logistic_prediction": value > 0.5,
        }


def test_grid_is_balanced_and_injection_text_is_shared() -> None:
    rows: list[dict[str, Any]] = build_grid(6, seed=1)
    assert len(rows) == 6 * len(FORMATS) * len(WRAPPERS) * 2
    frame: pd.DataFrame = pd.DataFrame(rows)
    assert frame.groupby(["format", "wrapper", "condition"]).size().nunique() == 1
    for _record_id, group in frame.groupby("record_id"):
        attacks: list[str] = [
            r.messages[-1]["content"] for r in group.itertuples() if r.condition == "attack"
        ]
        cleans: list[str] = [
            r.messages[-1]["content"] for r in group.itertuples() if r.condition == "clean"
        ]
        assert all(len(a) > len(c) for a, c in zip(attacks, cleans, strict=True))


def test_tool_result_wrapper_is_user_role_json() -> None:
    row: dict[str, Any] = next(r for r in build_grid(1) if r["wrapper"] == "tool_result")
    assert row["messages"][-1]["role"] == "user"
    assert row["messages"][-1]["content"].startswith('{"result"')


def test_summarize_detects_injection_and_format_shift() -> None:
    def capture(messages: list[dict[str, str]]) -> torch.Tensor:
        text: str = messages[-1]["content"]
        return torch.tensor([("PWNED" in text or "attacker" in text or "hidden" in text) * 1.0])

    scores: pd.DataFrame = score_grid(build_grid(6), capture, FakeScorer())
    summary: pd.DataFrame = summarize(scores)
    assert np.allclose(summary.auroc, 1.0)
    assert np.allclose(summary.clean_shift, 0.0)


def test_resume_drops_partial_tail_and_keeps_complete_rows(tmp_path: Path) -> None:
    sink: Path = tmp_path / "scores.jsonl"
    good: dict[str, Any] = {"record_id": 0, "format": "code", "wrapper": "inline", "condition": "clean"}
    sink.write_text(json.dumps(good) + "\n" + '{"record_id": 1, "form', encoding="utf-8")
    repair_tail(sink)
    assert sink.read_text(encoding="utf-8").endswith("\n")
    assert list(load_done(sink)) == [(0, "code", "inline", "clean")]


def test_resume_refuses_a_different_grid_identity(tmp_path: Path) -> None:
    path: Path = tmp_path / "identity.json"
    check_identity(path, {"n_records": 60, "seed": 0})
    check_identity(path, {"n_records": 60, "seed": 0})
    with pytest.raises(RuntimeError, match="new --output-dir"):
        check_identity(path, {"n_records": 30, "seed": 0})
