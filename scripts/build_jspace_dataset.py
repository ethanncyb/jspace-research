# Build paired clean/injected JSpace probe datasets (records CSVs).
#
# Label-by-construction design:
#   - label 1 ("injected"): benchmark instruction wrapped by an attacker
#     transformation from evolving_attacker.starter_strategies().
#   - label 0 ("clean"): (a) the bare benchmark instruction (direct request),
#     (b) benign user texts sampled from config.data.benign_prompts and the
#     benign prompt sets in data/evaluations/ + data/experiments/probe-swap.json.
#   - heldout_families.csv: dev instructions wrapped by the evaluation-only
#     held-out attack families — never used for training.
#
# All prompts are prefixed with evolution.baseline_prompt, matching the
# eval_harness records contract (baseline, prompt, label, category).
#
# Usage: python scripts/build_jspace_dataset.py [--config config.yaml] [--smoke]
from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Sequence
from pathlib import Path

from promptguard.benchmark_setup import BenchmarkExample, load_benchmark_splits
from promptguard.config import load_config
from promptguard.eval_harness import EvalRecord
from promptguard.evolving_attacker import (
    heldout_attack_strategies,
    starter_strategies,
)

BENIGN_PROMPT_GLOBS = (
    "data/evaluations/*.json",
    "data/experiments/probe-swap.json",
)


def load_benign_pool(config_prompts: Sequence[str]) -> list[str]:
    """Benign user texts: config list + prompts from the lens eval datasets."""

    pool = list(config_prompts)
    for pattern in BENIGN_PROMPT_GLOBS:
        for path in sorted(Path().glob(pattern)):
            try:
                items = json.loads(path.read_text(encoding="utf-8"))["items"]
            except (json.JSONDecodeError, KeyError):
                continue
            pool.extend(
                item["prompt"] for item in items if isinstance(item.get("prompt"), str)
            )
    if not pool:
        raise ValueError("benign prompt pool is empty")
    return pool


def build_partition_records(
    examples: Sequence[BenchmarkExample],
    *,
    baseline: str,
    strategies,
    benign_pool: Sequence[str],
    benign_ratio: float,
    rng: random.Random,
) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for example in examples:
        records.append(
            EvalRecord(
                baseline,
                f"{baseline}\n{example.instruction}",
                0,
                f"direct_request:{example.category}",
            )
        )
        for strategy in strategies:
            records.append(
                EvalRecord(
                    baseline,
                    f"{baseline}\n{strategy.transform(example.instruction)}",
                    1,
                    f"attack:{strategy.name}",
                )
            )
    n_injected = sum(record.label for record in records)
    n_benign = max(1, round(n_injected * benign_ratio))
    shuffled = list(benign_pool)
    rng.shuffle(shuffled)
    for index in range(n_benign):
        records.append(
            EvalRecord(
                baseline,
                f"{baseline}\n{shuffled[index % len(shuffled)]}",
                0,
                "benign_user",
            )
        )
    rng.shuffle(records)
    return records


def build_heldout_records(
    examples: Sequence[BenchmarkExample],
    *,
    baseline: str,
) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for example in examples:
        for strategy in heldout_attack_strategies():
            records.append(
                EvalRecord(
                    baseline,
                    f"{baseline}\n{strategy.transform(example.instruction)}",
                    1,
                    f"heldout:{strategy.name}",
                )
            )
        records.append(
            EvalRecord(
                baseline,
                f"{baseline}\n{example.instruction}",
                0,
                f"direct_request:{example.category}",
            )
        )
    return records


def write_records(records: Sequence[EvalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["baseline", "prompt", "label", "category"])
        for record in records:
            writer.writerow(
                [record.baseline, record.prompt, record.label, record.category]
            )


def summarize(name: str, records: Sequence[EvalRecord]) -> str:
    injected = sum(record.label for record in records)
    return f"{name}: {len(records)} records ({injected} injected, {len(records) - injected} clean)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="data/experiments/jspace")
    parser.add_argument("--benign-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 strategies x 4 examples per partition (fast end-to-end check)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    rng = random.Random(args.seed)
    split = load_benchmark_splits(config.benchmarks)
    baseline = config.evolution.baseline_prompt
    benign_pool = load_benign_pool(config.data.benign_prompts)

    strategies = starter_strategies()
    training, development, final = (
        split.training,
        split.attacker_development,
        split.final_test,
    )
    if args.smoke:
        strategies = strategies[:2]
        training, development, final = training[:4], development[:4], final[:4]

    output_dir = Path(args.output_dir)
    partitions = {
        "train": build_partition_records(
            training,
            baseline=baseline,
            strategies=strategies,
            benign_pool=benign_pool,
            benign_ratio=args.benign_ratio,
            rng=rng,
        ),
        "dev": build_partition_records(
            development,
            baseline=baseline,
            strategies=strategies,
            benign_pool=benign_pool,
            benign_ratio=args.benign_ratio,
            rng=rng,
        ),
        "final_test": build_partition_records(
            final,
            baseline=baseline,
            strategies=strategies,
            benign_pool=benign_pool,
            benign_ratio=args.benign_ratio,
            rng=rng,
        ),
        "heldout_families": build_heldout_records(development, baseline=baseline),
    }
    for name, records in partitions.items():
        write_records(records, output_dir / f"{name}.csv")
        print(summarize(name, records))
    print(f"wrote {len(partitions)} CSVs to {output_dir}")


if __name__ == "__main__":
    main()
