from __future__ import annotations

import json
from pathlib import Path

from gsm8k_jspace.config import load_config
from gsm8k_jspace.datasets.gsm8k import (
    clean_gsm8k_answer,
    example_id_for_index,
    load_gsm8k_examples,
)

FIXTURE = Path(__file__).parent / "fixtures" / "gsm8k_rows.json"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _rows():
    return json.loads(FIXTURE.read_text())


def test_clean_calculator_annotations():
    raw = "48/2 = <<48/2=24>>24.\n#### 24"
    assert clean_gsm8k_answer(raw) == "48/2 = 24.\n#### 24"


def test_stable_ids_and_first_subset():
    cfg = load_config(CONFIGS / "smoke.yaml")
    first = load_gsm8k_examples(cfg, rows=_rows())
    second = load_gsm8k_examples(cfg, rows=_rows())
    assert [row.example_id for row in first] == [row.example_id for row in second]
    assert first[0].example_id == example_id_for_index(0, "test")
    assert first[0].gold_answer == "42"
    assert len(first) == 5


def test_shuffled_is_deterministic():
    cfg = load_config(CONFIGS / "smoke.yaml")
    cfg.benchmark.selection = "shuffled"
    cfg.benchmark.selection_seed = 4
    cfg.benchmark.subset_size = 3
    a = load_gsm8k_examples(cfg, rows=_rows())
    b = load_gsm8k_examples(cfg, rows=_rows())
    assert [row.example_id for row in a] == [row.example_id for row in b]
    assert len(a) == 3
