from gsm8k_jspace.benchmarks import get_benchmark
from gsm8k_jspace.config import AppConfig
from gsm8k_jspace.datasets.gsm8k import (
    example_id_for_index,
    extract_gold_answer,
    load_gsm8k_examples,
)
from gsm8k_jspace.datasets.selection import selection_records
from gsm8k_jspace.types import GSM8KExample


def load_examples(cfg: AppConfig, **kwargs) -> list[GSM8KExample]:
    return get_benchmark(cfg).load_examples(cfg, **kwargs)


__all__ = [
    "example_id_for_index",
    "extract_gold_answer",
    "load_examples",
    "load_gsm8k_examples",
    "selection_records",
]
