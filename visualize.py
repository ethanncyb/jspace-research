"""Presentation-ready charts for Promptguard CSV evaluation artifacts.

This module is intentionally read-only with respect to experiment data: it
consumes the CSV files written by eval_harness.py, evolution_loop.py, and
benchmark_runner.py and writes PNG charts beneath ``outputs/charts``.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.switch_backend("Agg")

BENIGN = "#2A9D8F"
BENIGN_ALT = "#277DA1"
ATTACK = "#E76F51"
ATTACK_ALT = "#F4A261"
THRESHOLD = "#6C757D"
GRID = "#DDE2E5"

DEFAULT_EVAL = Path("outputs/eval/examples.csv")
DEFAULT_ROUNDS = Path("outputs/evolution/round_metrics.csv")
DEFAULT_ATTACKS = Path("outputs/evolution/attacks.csv")
DEFAULT_EVOLUTION_CROSS = Path("outputs/evolution/cross_stage.csv")
DEFAULT_BENCHMARK_ASR = Path("outputs/benchmarks/asr_aqc.csv")
DEFAULT_BENCHMARK_CROSS = Path("outputs/benchmarks/cross_stage.csv")
DEFAULT_CHART_DIR = Path("outputs/charts")


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "axes.edgecolor": GRID,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _read_csv(path: str | Path, required: set[str]) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"required visualization input not found: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source} is missing CSV columns: {sorted(missing)}")
        return list(reader)


def _save(fig: plt.Figure, output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, facecolor="white")
    plt.close(fig)
    return destination


def _layer_columns(rows: list[dict[str, str]]) -> list[tuple[int, str]]:
    if not rows:
        raise ValueError("evaluation CSV contains no examples")
    found = []
    for column in rows[0]:
        match = re.fullmatch(r"layer_(\d+)_score", column)
        if match:
            found.append((int(match.group(1)), column))
    if not found:
        raise ValueError("evaluation CSV contains no layer_<index>_score columns")
    return sorted(found)


def _is_attack(row: dict[str, str]) -> bool:
    label = row.get("label", "").strip().lower()
    if label in {"1", "true", "unsafe", "attack", "malicious"}:
        return True
    return row.get("category", "").strip().lower() != "benign"


def plot_layer_drift_lines(
    eval_csv: str | Path = DEFAULT_EVAL,
    output: str | Path = DEFAULT_CHART_DIR / "layer_drift_lines.png",
) -> Path:
    """Overlay benign and attack drift trajectories across hooked layers."""

    _style()
    rows = _read_csv(eval_csv, {"label", "category"})
    layers = _layer_columns(rows)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    grouped: dict[bool, list[list[float]]] = {False: [], True: []}
    for row in rows:
        attack = _is_attack(row)
        values = [float(row[column]) for _, column in layers]
        grouped[attack].append(values)
        ax.plot(
            [layer for layer, _ in layers],
            values,
            color=ATTACK if attack else BENIGN,
            alpha=0.18,
            linewidth=1.2,
        )
    for attack, label in [(False, "Benign mean"), (True, "Attack mean")]:
        if grouped[attack]:
            mean = np.asarray(grouped[attack], dtype=float).mean(axis=0)
            ax.plot(
                [layer for layer, _ in layers],
                mean,
                color=ATTACK if attack else BENIGN,
                marker="o",
                linewidth=3,
                label=label,
            )
    ax.set(title="Layerwise Drift: Benign vs. Attack Prompts", xlabel="Hooked layer index", ylabel="Drift score")
    ax.set_xticks([layer for layer, _ in layers])
    ax.set_ylim(0, 1)
    ax.legend(frameon=True)
    return _save(fig, output)


def plot_layer_drift_heatmap(
    eval_csv: str | Path = DEFAULT_EVAL,
    output: str | Path = DEFAULT_CHART_DIR / "layer_drift_heatmap.png",
) -> Path:
    """Render prompt-by-layer drift intensity, ordered benign then attack."""

    _style()
    rows = _read_csv(eval_csv, {"label", "category"})
    layers = _layer_columns(rows)
    ordered = sorted(enumerate(rows), key=lambda item: (_is_attack(item[1]), item[0]))
    matrix = np.asarray(
        [[float(row[column]) for _, column in layers] for _, row in ordered],
        dtype=float,
    )
    labels = [
        f"{index}: {'attack' if _is_attack(row) else 'benign'}"
        for index, row in ordered
    ]
    height = max(5.5, min(14, 0.42 * len(rows) + 2.5))
    fig, ax = plt.subplots(figsize=(11, height))
    sns.heatmap(
        matrix,
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        vmin=0,
        vmax=1,
        xticklabels=[layer for layer, _ in layers],
        yticklabels=labels,
        cbar_kws={"label": "Drift score"},
        linewidths=0.35,
        linecolor="white",
        ax=ax,
    )
    ax.set(title="Per-Prompt Layerwise Drift Heatmap", xlabel="Hooked layer index", ylabel="Evaluation prompt")
    return _save(fig, output)


def _method_label(method: str) -> str:
    normalized = method.lower().replace("-", "_")
    if "evolved" in normalized or "evolving" in normalized:
        return "Evolving attacker"
    if "renellm" in normalized:
        return "ReNeLLM"
    if normalized == "tap" or "tap" in normalized:
        return "TAP"
    return method.replace("_", " ").title()


def plot_asr_over_rounds(
    benchmark_csv: str | Path = DEFAULT_BENCHMARK_ASR,
    output: str | Path = DEFAULT_CHART_DIR / "asr_over_rounds.png",
    round_metrics_csv: str | Path = DEFAULT_ROUNDS,
) -> Path:
    """Compare evolving and baseline attack success rates by guard round."""

    _style()
    series: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    benchmark = Path(benchmark_csv)
    if benchmark.exists():
        rows = _read_csv(benchmark, {"guard_round", "method", "asr"})
        aggregate = [row for row in rows if row.get("source_dataset") == "all"]
        for row in aggregate or rows:
            guard_round = int(row["guard_round"])
            if guard_round >= 1:
                series[_method_label(row["method"])][guard_round].append(float(row["asr"]))
    rounds_path = Path(round_metrics_csv)
    if "Evolving attacker" not in series and rounds_path.exists():
        rows = _read_csv(rounds_path, {"round", "attacker_asr"})
        for row in rows:
            series["Evolving attacker"][int(row["round"])].append(float(row["attacker_asr"]))
    if not series:
        raise FileNotFoundError(f"no ASR input found at {benchmark} or {rounds_path}")
    colors = {"Evolving attacker": ATTACK, "ReNeLLM": ATTACK_ALT, "TAP": "#9C6644"}
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for method, by_round in sorted(series.items(), key=lambda item: item[0] != "Evolving attacker"):
        xs = sorted(by_round)
        ys = [float(np.mean(by_round[x])) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.7, label=method, color=colors.get(method))
    ax.set(title="Attack Success Rate over Evolution Rounds", xlabel="Evolution / guard round", ylabel="Attack success rate (ASR)")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted({x for values in series.values() for x in values}))
    ax.legend(frameon=True)
    return _save(fig, output)


def plot_guard_robustness(
    round_metrics_csv: str | Path = DEFAULT_ROUNDS,
    output: str | Path = DEFAULT_CHART_DIR / "guard_robustness.png",
) -> Path:
    """Plot unsafe recall and benign pass rate on a shared round axis."""

    _style()
    rows = _read_csv(round_metrics_csv, {"round", "probe_unsafe_recall", "probe_benign_pass_rate"})
    rows.sort(key=lambda row: int(row["round"]))
    xs = [int(row["round"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.plot(xs, [float(row["probe_unsafe_recall"]) for row in rows], color=ATTACK, marker="o", linewidth=2.8, label="Unsafe recall")
    ax.plot(xs, [float(row["probe_benign_pass_rate"]) for row in rows], color=BENIGN, marker="s", linestyle="--", linewidth=2.8, label="Benign pass rate")
    ax.set(title="Guard Robustness over Evolution Rounds", xlabel="Evolution round", ylabel="Rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.legend(frameon=True)
    return _save(fig, output)


def plot_cross_stage_matrix(
    cross_stage_csv: str | Path = DEFAULT_BENCHMARK_CROSS,
    output: str | Path = DEFAULT_CHART_DIR / "cross_stage_matrix.png",
    fallback_csv: str | Path = DEFAULT_EVOLUTION_CROSS,
) -> Path:
    """Render evolved Attack-K by Guard-J ASR cross-play."""

    _style()
    source = Path(cross_stage_csv)
    if not source.exists():
        source = Path(fallback_csv)
    rows = _read_csv(source, {"attack_round", "guard_round", "asr"})
    if "source_dataset" in rows[0]:
        aggregate = [row for row in rows if row.get("source_dataset") == "all"]
        rows = aggregate or rows
    rows = [
        row
        for row in rows
        if row["attack_round"].strip().isdigit()
        and row.get("attack_family", "evolved") != "heldout_untrained"
    ]
    if not rows:
        raise ValueError(f"{source} contains no evolved Attack-K rows")
    attack_rounds = sorted({int(row["attack_round"]) for row in rows})
    guard_rounds = sorted({int(row["guard_round"]) for row in rows})
    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        buckets[(int(row["attack_round"]), int(row["guard_round"]))].append(float(row["asr"]))
    matrix = np.full((len(attack_rounds), len(guard_rounds)), np.nan)
    for i, attack_round in enumerate(attack_rounds):
        for j, guard_round in enumerate(guard_rounds):
            values = buckets[(attack_round, guard_round)]
            if values:
                matrix[i, j] = float(np.mean(values))
    fig, ax = plt.subplots(figsize=(max(7, len(guard_rounds) * 1.15 + 3), max(5.5, len(attack_rounds) * 0.8 + 3)))
    sns.heatmap(matrix, cmap="YlOrRd", vmin=0, vmax=1, annot=True, fmt=".2f", mask=np.isnan(matrix), xticklabels=[f"Guard-{j}" for j in guard_rounds], yticklabels=[f"Attack-{k}" for k in attack_rounds], cbar_kws={"label": "ASR"}, ax=ax)
    ax.set(title="Cross-Stage Attack–Guard Generalization", xlabel="Guard checkpoint", ylabel="Attacker checkpoint")
    return _save(fig, output)


def plot_strategy_pool_composition(
    attacks_csv: str | Path = DEFAULT_ATTACKS,
    output: str | Path = DEFAULT_CHART_DIR / "strategy_pool_composition.png",
) -> Path:
    """Plot per-round strategy selection frequency as normalized stacked bars."""

    _style()
    rows = _read_csv(attacks_csv, {"round", "strategy"})
    rounds = sorted({int(row["round"]) for row in rows})
    strategies = sorted({row["strategy"] for row in rows})
    counts = {round_index: Counter(row["strategy"] for row in rows if int(row["round"]) == round_index) for round_index in rounds}
    totals = {round_index: max(sum(counts[round_index].values()), 1) for round_index in rounds}
    palette = sns.color_palette("tab20", n_colors=max(len(strategies), 1))
    fig, ax = plt.subplots(figsize=(max(10.5, 1.1 * len(rounds) + 5), 7))
    bottoms = np.zeros(len(rounds))
    for color, strategy in zip(palette, strategies, strict=True):
        values = np.asarray([counts[r][strategy] / totals[r] for r in rounds])
        ax.bar(rounds, values, bottom=bottoms, label=strategy, color=color, width=0.72)
        bottoms += values
    ax.set(title="Strategy Pool Selection Pressure over Time", xlabel="Evolution round", ylabel="Share of strategy selections")
    ax.set_ylim(0, 1)
    ax.set_xticks(rounds)
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize="small")
    return _save(fig, output)


def plot_intervention_triggers(
    eval_csv: str | Path = DEFAULT_EVAL,
    output: str | Path = DEFAULT_CHART_DIR / "intervention_triggers.png",
    *,
    threshold: float = 0.5,
) -> Path:
    """Show threshold crossing trajectories for available intervention modes."""

    _style()
    rows = _read_csv(eval_csv, {"intervention_triggered", "intervention_mode"})
    layers = _layer_columns(rows)
    triggered = [row for row in rows if row["intervention_triggered"].strip().lower() in {"1", "true", "yes"}]
    if not triggered:
        raise ValueError(f"{eval_csv} contains no triggered intervention examples")
    selected = []
    for mode in ("circuit_breaker", "hard_stop"):
        match = next((row for row in triggered if row["intervention_mode"] == mode), None)
        if match is not None:
            selected.append(match)
    if not selected:
        selected = [triggered[0]]
    fig, axes = plt.subplots(1, len(selected), figsize=(7 * len(selected), 5.8), squeeze=False, sharey=True)
    xs = [layer for layer, _ in layers]
    for ax, row in zip(axes[0], selected, strict=True):
        values = [float(row[column]) for _, column in layers]
        crossing = next((index for index, value in enumerate(values) if value >= threshold), None)
        ax.plot(xs, values, color=ATTACK, marker="o", linewidth=2.8, label="Layer drift")
        ax.axhline(threshold, color=THRESHOLD, linestyle="--", linewidth=2, label=f"Threshold ({threshold:.2f})")
        if crossing is not None:
            ax.scatter(xs[crossing], values[crossing], s=150, color=ATTACK_ALT, edgecolor="black", zorder=5, label=f"First crossing: layer {xs[crossing]}")
            ax.axvline(xs[crossing], color=ATTACK_ALT, linestyle=":", alpha=0.8)
        ax.set(title=row["intervention_mode"].replace("_", " ").title(), xlabel="Hooked layer index")
        ax.set_xticks(xs)
        ax.legend(frameon=True, fontsize="small")
    axes[0, 0].set_ylabel("Drift score")
    axes[0, 0].set_ylim(0, 1)
    fig.suptitle("Intervention Trigger Trajectory", fontweight="bold", y=1.03)
    return _save(fig, output)


def plot_heldout_generalization(
    cross_stage_csv: str | Path = DEFAULT_EVOLUTION_CROSS,
    output: str | Path = DEFAULT_CHART_DIR / "heldout_generalization.png",
) -> Path:
    """Plot held-out, never-trained attack-family ASR against each guard."""

    _style()
    rows = _read_csv(cross_stage_csv, {"attack_round", "guard_round", "asr"})
    heldout = [row for row in rows if row["attack_round"].lower() == "heldout" or row.get("attack_family") == "heldout_untrained"]
    if not heldout:
        raise ValueError(f"{cross_stage_csv} contains no held-out attack-family rows")
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in heldout:
        buckets[int(row["guard_round"])].append(float(row["asr"]))
    xs = sorted(buckets)
    ys = [float(np.mean(buckets[x])) for x in xs]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.plot(xs, ys, color=ATTACK, marker="o", markersize=8, linewidth=3, label="Held-out attack family")
    ax.fill_between(xs, ys, alpha=0.12, color=ATTACK)
    ax.set(title="Held-Out Attack Generalization over Guard Rounds", xlabel="Guard checkpoint round", ylabel="Held-out attack success rate (ASR)")
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.legend(frameon=True)
    return _save(fig, output)


def generate_all_visuals(
    *,
    eval_csv: str | Path = DEFAULT_EVAL,
    round_metrics_csv: str | Path = DEFAULT_ROUNDS,
    attacks_csv: str | Path = DEFAULT_ATTACKS,
    evolution_cross_csv: str | Path = DEFAULT_EVOLUTION_CROSS,
    benchmark_asr_csv: str | Path = DEFAULT_BENCHMARK_ASR,
    benchmark_cross_csv: str | Path = DEFAULT_BENCHMARK_CROSS,
    output_dir: str | Path = DEFAULT_CHART_DIR,
    threshold: float = 0.5,
) -> list[Path]:
    """Generate the complete visualization suite and return written paths."""

    destination = Path(output_dir)
    return [
        plot_layer_drift_lines(eval_csv, destination / "layer_drift_lines.png"),
        plot_layer_drift_heatmap(eval_csv, destination / "layer_drift_heatmap.png"),
        plot_asr_over_rounds(benchmark_asr_csv, destination / "asr_over_rounds.png", round_metrics_csv),
        plot_guard_robustness(round_metrics_csv, destination / "guard_robustness.png"),
        plot_cross_stage_matrix(benchmark_cross_csv, destination / "cross_stage_matrix.png", evolution_cross_csv),
        plot_strategy_pool_composition(attacks_csv, destination / "strategy_pool_composition.png"),
        plot_intervention_triggers(eval_csv, destination / "intervention_triggers.png", threshold=threshold),
        plot_heldout_generalization(evolution_cross_csv, destination / "heldout_generalization.png"),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="generate every chart")
    selection.add_argument(
        "--chart",
        choices=[
            "layer_drift_lines",
            "layer_drift_heatmap",
            "asr_over_rounds",
            "guard_robustness",
            "cross_stage_matrix",
            "strategy_pool_composition",
            "intervention_triggers",
            "heldout_generalization",
        ],
        help="generate one chart",
    )
    parser.add_argument("--eval-csv", default=str(DEFAULT_EVAL))
    parser.add_argument("--round-metrics-csv", default=str(DEFAULT_ROUNDS))
    parser.add_argument("--attacks-csv", default=str(DEFAULT_ATTACKS))
    parser.add_argument("--evolution-cross-csv", default=str(DEFAULT_EVOLUTION_CROSS))
    parser.add_argument("--benchmark-asr-csv", default=str(DEFAULT_BENCHMARK_ASR))
    parser.add_argument("--benchmark-cross-csv", default=str(DEFAULT_BENCHMARK_CROSS))
    parser.add_argument("--output-dir", default=str(DEFAULT_CHART_DIR))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    common = {
        "eval_csv": args.eval_csv,
        "round_metrics_csv": args.round_metrics_csv,
        "attacks_csv": args.attacks_csv,
        "evolution_cross_csv": args.evolution_cross_csv,
        "benchmark_asr_csv": args.benchmark_asr_csv,
        "benchmark_cross_csv": args.benchmark_cross_csv,
        "output_dir": args.output_dir,
        "threshold": args.threshold,
    }
    if args.all:
        paths = generate_all_visuals(**common)
    else:
        output = Path(args.output_dir) / f"{args.chart}.png"
        functions: dict[str, Callable[[], Path]] = {
            "layer_drift_lines": lambda: plot_layer_drift_lines(args.eval_csv, output),
            "layer_drift_heatmap": lambda: plot_layer_drift_heatmap(args.eval_csv, output),
            "asr_over_rounds": lambda: plot_asr_over_rounds(args.benchmark_asr_csv, output, args.round_metrics_csv),
            "guard_robustness": lambda: plot_guard_robustness(args.round_metrics_csv, output),
            "cross_stage_matrix": lambda: plot_cross_stage_matrix(args.benchmark_cross_csv, output, args.evolution_cross_csv),
            "strategy_pool_composition": lambda: plot_strategy_pool_composition(args.attacks_csv, output),
            "intervention_triggers": lambda: plot_intervention_triggers(args.eval_csv, output, threshold=args.threshold),
            "heldout_generalization": lambda: plot_heldout_generalization(args.evolution_cross_csv, output),
        }
        paths = [functions[args.chart]()]
    print("Generated charts:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
