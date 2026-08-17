"""Download, normalize, subset, and split HarmBench/AdvBench behaviors."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from promptguard.config import BenchmarkConfig


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    source_dataset: str
    category: str
    instruction: str


@dataclass(frozen=True)
class BenchmarkSplit:
    train: tuple[BenchmarkExample, ...]
    heldout: tuple[BenchmarkExample, ...]
    heldout_categories: tuple[str, ...]


def _materialize(source: str | Path, cache_dir: str | Path) -> Path:
    candidate = Path(source)
    if candidate.exists():
        return candidate
    parsed = urlparse(str(source))
    if parsed.scheme not in {"http", "https"}:
        raise FileNotFoundError(source)
    destination = Path(cache_dir) / Path(parsed.path).name
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(str(source), timeout=30)
    response.raise_for_status()
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    return destination


def _choose_subset(
    examples: Sequence[BenchmarkExample], size: int | str, *, seed: int
) -> list[BenchmarkExample]:
    if isinstance(size, str):
        if size.lower() != "full":
            raise ValueError("benchmark subset size must be an integer or 'full'")
        return list(examples)
    if size <= 0:
        raise ValueError("benchmark subset size must be positive")
    if size >= len(examples):
        return list(examples)

    # Round-robin over shuffled categories prevents the POC subset from being
    # accidentally dominated by one HarmBench semantic category.
    rng = random.Random(seed)
    groups: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        groups[example.category].append(example)
    for group in groups.values():
        rng.shuffle(group)
    categories = sorted(groups)
    rng.shuffle(categories)
    selected: list[BenchmarkExample] = []
    while len(selected) < size and categories:
        remaining = []
        for category in categories:
            if groups[category] and len(selected) < size:
                selected.append(groups[category].pop())
            if groups[category]:
                remaining.append(category)
        categories = remaining
    return selected


def load_harmbench(
    source: str | Path,
    *,
    cache_dir: str | Path,
    subset_size: int | str = 50,
    seed: int = 7,
) -> list[BenchmarkExample]:
    path = _materialize(source, cache_dir)
    examples: list[BenchmarkExample] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            instruction = (row.get("Behavior") or row.get("behavior") or "").strip()
            if not instruction:
                continue
            identifier = (
                row.get("BehaviorID") or row.get("behavior_id") or f"harmbench-{index}"
            )
            category = (
                row.get("SemanticCategory")
                or row.get("FunctionalCategory")
                or row.get("category")
                or "uncategorized"
            )
            examples.append(
                BenchmarkExample(
                    str(identifier), "harmbench", str(category), instruction
                )
            )
    if not examples:
        raise ValueError(f"no HarmBench behaviors found in {path}")
    return _choose_subset(examples, subset_size, seed=seed)


def load_advbench(
    source: str | Path,
    *,
    cache_dir: str | Path,
    subset_size: int | str = 50,
    seed: int = 7,
) -> list[BenchmarkExample]:
    path = _materialize(source, cache_dir)
    examples: list[BenchmarkExample] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            instruction = (
                row.get("goal") or row.get("instruction") or row.get("Behavior") or ""
            ).strip()
            if not instruction:
                continue
            category = row.get("category") or "advbench_harmful_behavior"
            examples.append(
                BenchmarkExample(
                    f"advbench-{index}", "advbench", str(category), instruction
                )
            )
    if not examples:
        raise ValueError(f"no AdvBench behaviors found in {path}")
    return _choose_subset(examples, subset_size, seed=seed)


def load_benchmarks(config: BenchmarkConfig) -> list[BenchmarkExample]:
    return [
        *load_harmbench(
            config.harmbench_url,
            cache_dir=config.cache_dir,
            subset_size=config.harmbench_subset_size,
            seed=config.seed,
        ),
        *load_advbench(
            config.advbench_url,
            cache_dir=config.cache_dir,
            subset_size=config.advbench_subset_size,
            seed=config.seed + 1,
        ),
    ]


def heldout_split(
    examples: Iterable[BenchmarkExample],
    *,
    heldout_categories: Sequence[str] = (),
    heldout_fraction: float = 0.2,
    seed: int = 7,
) -> BenchmarkSplit:
    """Reserve entire categories when possible, then instructions if needed."""

    values = list(examples)
    if not 0 <= heldout_fraction < 1:
        raise ValueError("heldout_fraction must be in [0,1)")
    requested = set(heldout_categories)
    categories = sorted({example.category for example in values})
    unknown = requested - set(categories)
    if unknown:
        raise ValueError(f"unknown held-out categories: {sorted(unknown)}")

    if not requested and len(categories) > 1 and heldout_fraction > 0:
        rng = random.Random(seed)
        shuffled = list(categories)
        rng.shuffle(shuffled)
        count = max(1, round(len(categories) * heldout_fraction))
        requested.update(shuffled[:count])

    heldout = [example for example in values if example.category in requested]
    train = [example for example in values if example.category not in requested]
    if not heldout and heldout_fraction > 0 and values:
        # AdvBench's canonical CSV has no categories. Reserve instruction IDs
        # deterministically so the same behaviors never enter attacker training.
        rng = random.Random(seed)
        shuffled = list(values)
        rng.shuffle(shuffled)
        count = max(1, round(len(values) * heldout_fraction))
        heldout_ids = {example.id for example in shuffled[:count]}
        heldout = [example for example in values if example.id in heldout_ids]
        train = [example for example in values if example.id not in heldout_ids]
    return BenchmarkSplit(tuple(train), tuple(heldout), tuple(sorted(requested)))
