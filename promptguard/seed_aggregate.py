"""Aggregate held-out evolution ASR across independently seeded runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

plt.switch_backend("Agg")


@dataclass(frozen=True)
class SeedAggregateRow:
    round: int
    run_count: int
    mean_asr: float
    std_asr: float
    lower_asr: float
    upper_asr: float


def _read_run(run_dir: str | Path) -> tuple[str, dict[int, float]]:
    directory = Path(run_dir)
    metadata_path = directory / "run_metadata.json"
    label = directory.name
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        label = f"seed {metadata['seed']}"
    metrics_path = directory / "round_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"round", "heldout_attacker_asr"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{metrics_path} is missing columns: {sorted(missing)}")
        values = {
            int(row["round"]): float(row["heldout_attacker_asr"])
            for row in reader
        }
    if not values:
        raise ValueError(f"{metrics_path} contains no rounds")
    return label, values


def aggregate_heldout_asr(
    run_dirs: Sequence[str | Path],
    *,
    csv_path: str | Path,
    chart_path: str | Path,
) -> list[SeedAggregateRow]:
    """Write per-round mean and sample standard deviation plus a band chart."""

    if len(run_dirs) < 2:
        raise ValueError("at least two seeded run directories are required")
    runs = [_read_run(run_dir) for run_dir in run_dirs]
    expected_rounds = set(runs[0][1])
    for label, values in runs[1:]:
        if set(values) != expected_rounds:
            raise ValueError(f"{label} has a different set of rounds")

    rows: list[SeedAggregateRow] = []
    for round_index in sorted(expected_rounds):
        values = [series[round_index] for _, series in runs]
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        rows.append(
            SeedAggregateRow(
                round=round_index,
                run_count=len(values),
                mean_asr=mean,
                std_asr=std,
                lower_asr=max(0.0, mean - std),
                upper_asr=min(1.0, mean + std),
            )
        )

    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    figure_path = Path(chart_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    rounds = [row.round for row in rows]
    means = [row.mean_asr for row in rows]
    lowers = [row.lower_asr for row in rows]
    uppers = [row.upper_asr for row in rows]
    for label, series in runs:
        ax.plot(
            rounds,
            [series[round_index] for round_index in rounds],
            color="#8D99AE",
            alpha=0.35,
            linewidth=1.4,
            marker="o",
            markersize=3,
            label=label,
        )
    ax.fill_between(
        rounds,
        lowers,
        uppers,
        color="#E76F51",
        alpha=0.22,
        label="Mean ± 1 sample SD",
    )
    ax.plot(
        rounds,
        means,
        color="#D1495B",
        marker="o",
        linewidth=3,
        label="Mean held-out ASR",
    )
    ax.set(
        title="Held-out ASR Across Random Seeds",
        xlabel="Evolution round",
        ylabel="Held-out attack success rate",
        xticks=rounds,
        ylim=(0, max(0.08, min(1.0, max(uppers) * 1.25))),
    )
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="seeded evolution output directories")
    parser.add_argument("--output-dir", default="outputs/evolution_multiseed")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    rows = aggregate_heldout_asr(
        args.run_dirs,
        csv_path=output_dir / "heldout_asr_mean_std.csv",
        chart_path=output_dir / "heldout_asr_mean_std.png",
    )
    print(json.dumps([asdict(row) for row in rows], indent=2))


if __name__ == "__main__":
    main()
