from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsm8k_jspace.analysis.loaders import iter_capture_rows
from gsm8k_jspace.analysis.tables import (
    token_layer_presence_table,
    token_layer_rank_grid,
    topk_by_layer_table,
)


def _write_capture(run_dir: Path, example_id: str, rows: list[dict]) -> None:
    cap_dir = run_dir / "captures"
    cap_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    with gzip.open(cap_dir / f"{example_id}.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(payload)


def _run(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(run_dir=tmp_path)


def test_topk_token_layers(tmp_path: Path):
    rows = [
        {
            "example_id": "gsm8k_test_000000",
            "layer": 10,
            "absolute_position": 4,
            "generated_position": 3,
            "token_text": " 18",
            "top_jspace_tokens": [
                {"token_id": 18, "text": " 18", "logit": 10.0},
                {"token_id": 2, "text": " the", "logit": 5.0},
            ],
        },
        {
            "example_id": "gsm8k_test_000000",
            "layer": 18,
            "absolute_position": 4,
            "generated_position": 3,
            "token_text": " 18",
            "top_jspace_tokens": [
                {"token_id": 18, "text": " 18", "logit": 12.0},
                {"token_id": 7, "text": " ans", "logit": 7.0},
            ],
        },
        {
            "example_id": "gsm8k_test_000000",
            "layer": 10,
            "absolute_position": 1,
            "generated_position": 0,
            "token_text": "The",
            "top_jspace_tokens": [
                {"token_id": 99, "text": " ignored", "logit": 99.0},
            ],
        },
    ]
    _write_capture(tmp_path, "gsm8k_test_000000", rows)
    run = _run(tmp_path)

    by_layer = topk_by_layer_table(
        run, example_id="gsm8k_test_000000", position="last", max_rank=2
    )
    assert [row["layer"] for row in by_layer] == [10, 18]
    assert by_layer[0]["rank_1"].startswith('" 18"')
    assert by_layer[1]["rank_2"].startswith('" ans"')

    presence = token_layer_presence_table(
        run, example_id="gsm8k_test_000000", position="last", max_rank=2
    )
    by_token = {row["token"]: row for row in presence}
    assert by_token['" 18"']["layers"] == "10, 18"
    assert by_token['" 18"']["best_rank"] == 1
    assert by_token['" the"']["layers"] == "10"
    assert by_token['" ans"']["layers"] == "18"

    grid = token_layer_rank_grid(
        run, example_id="gsm8k_test_000000", position="last", max_rank=2
    )
    eighteen = next(row for row in grid if row["token"] == '" 18"')
    assert eighteen["L10"] == 1
    assert eighteen["L18"] == 1
    the = next(row for row in grid if row["token"] == '" the"')
    assert the["L10"] == 2
    assert the["L18"] is None


def test_plot_token_layer_heatmap(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from gsm8k_jspace.analysis.plots import plot_token_layer_heatmap

    path = tmp_path / "topk.png"
    fig = plot_token_layer_heatmap(
        [
            {"token": '" 18"', "L10": 1, "L18": 1},
            {"token": '" the"', "L10": 2, "L18": None},
        ],
        path=path,
    )
    fig.clf()
    assert path.exists()


def test_iter_capture_rows_filters_example(tmp_path: Path):
    _write_capture(
        tmp_path,
        "gsm8k_test_000000",
        [{"example_id": "gsm8k_test_000000", "layer": 1, "absolute_position": 0}],
    )
    _write_capture(
        tmp_path,
        "gsm8k_test_000001",
        [{"example_id": "gsm8k_test_000001", "layer": 2, "absolute_position": 0}],
    )
    ids = [row["example_id"] for row in iter_capture_rows(tmp_path, example_id="gsm8k_test_000001")]
    assert ids == ["gsm8k_test_000001"]
