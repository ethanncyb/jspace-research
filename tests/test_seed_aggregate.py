from __future__ import annotations

import csv
import json

import pytest

from promptguard.seed_aggregate import aggregate_heldout_asr


def _write_run(path, seed, values):
    path.mkdir()
    (path / "run_metadata.json").write_text(json.dumps({"seed": seed}))
    with (path / "round_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "heldout_attacker_asr"])
        writer.writeheader()
        for round_index, value in enumerate(values, start=1):
            writer.writerow({"round": round_index, "heldout_attacker_asr": value})


def test_aggregate_heldout_asr_writes_sample_std_and_chart(tmp_path):
    run_dirs = [tmp_path / f"seed_{seed}" for seed in (1, 2, 3)]
    _write_run(run_dirs[0], 1, [0.0, 0.1])
    _write_run(run_dirs[1], 2, [0.1, 0.2])
    _write_run(run_dirs[2], 3, [0.2, 0.3])
    rows = aggregate_heldout_asr(
        run_dirs,
        csv_path=tmp_path / "aggregate.csv",
        chart_path=tmp_path / "aggregate.png",
    )
    assert rows[0].mean_asr == pytest.approx(0.1)
    assert rows[0].std_asr == pytest.approx(0.1)
    assert (tmp_path / "aggregate.csv").exists()
    assert (tmp_path / "aggregate.png").stat().st_size > 0
