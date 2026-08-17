"""Standalone layerwise diagnostic for semantically encoded attack prompts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from promptguard.benchmark_setup import BenchmarkExample, heldout_split, load_benchmarks
from promptguard.config import ResearchConfig, load_config
from promptguard.drift_probe import DeltaExample, DriftProbe, extract_appended_delta
from promptguard.evolving_attacker import (
    heldout_acrostic_wrapper,
    heldout_translation_wrapper,
)
from promptguard.model_hooks import HookedModel, load_model

plt.switch_backend("Agg")

DeltaExtractor = Callable[[str, str], DeltaExample]


@dataclass(frozen=True)
class DiagnosticPrompt:
    example_id: str
    payload_id: str
    source_dataset: str
    semantic_category: str
    attack_family: str
    prompt: str


@dataclass(frozen=True)
class LayerScoreRow:
    example_id: str
    payload_id: str
    source_dataset: str
    semantic_category: str
    attack_family: str
    layer: int
    drift_score: float
    predicted_unsafe: int
    delta_l2_norm: float
    aggregate_drift_score: float
    aggregate_predicted_unsafe: int
    prompt: str


@dataclass(frozen=True)
class LayerSummaryRow:
    attack_family: str
    layer: int
    count: int
    mean_drift_score: float
    sample_std_drift_score: float
    detection_rate: float
    mean_delta_l2_norm: float
    sample_std_delta_l2_norm: float


TRANSFORMS = {
    "translated": heldout_translation_wrapper,
    "acrostic_encoding": heldout_acrostic_wrapper,
    "raw_control": lambda payload: payload,
}


def benchmark_payloads(
    config: ResearchConfig, *, split: str
) -> list[BenchmarkExample]:
    benchmark_split = heldout_split(
        load_benchmarks(config.benchmarks),
        heldout_categories=config.benchmarks.heldout_categories,
        heldout_fraction=config.benchmarks.heldout_fraction,
        seed=config.benchmarks.seed,
    )
    if split == "heldout":
        return list(benchmark_split.heldout)
    if split == "active":
        return list(benchmark_split.train)
    return [*benchmark_split.train, *benchmark_split.heldout]


def csv_payloads(path: str | Path) -> list[BenchmarkExample]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "instruction" not in (reader.fieldnames or []):
            raise ValueError("diagnostic dataset CSV requires an instruction column")
        return [
            BenchmarkExample(
                id=row.get("id") or f"row-{index}",
                source_dataset=row.get("source_dataset") or "custom",
                category=row.get("semantic_category")
                or row.get("category")
                or "uncategorized",
                instruction=row["instruction"],
            )
            for index, row in enumerate(reader)
            if row["instruction"].strip()
        ]


def build_diagnostic_prompts(
    payloads: Sequence[BenchmarkExample],
    *,
    families: Sequence[str] = ("translated", "acrostic_encoding"),
    raw_control: bool = True,
) -> list[DiagnosticPrompt]:
    requested = list(families)
    if raw_control:
        requested.insert(0, "raw_control")
    unknown = set(requested) - set(TRANSFORMS)
    if unknown:
        raise ValueError(f"unknown diagnostic families: {sorted(unknown)}")
    return [
        DiagnosticPrompt(
            example_id=f"{payload.id}:{family}",
            payload_id=payload.id,
            source_dataset=payload.source_dataset,
            semantic_category=payload.category,
            attack_family=family,
            prompt=TRANSFORMS[family](payload.instruction),
        )
        for payload in payloads
        for family in requested
    ]


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def diagnose_layers(
    model: HookedModel,
    probe: DriftProbe,
    prompts: Sequence[DiagnosticPrompt],
    *,
    baseline: str,
    pooling: str,
    threshold: float,
    extractor: DeltaExtractor | None = None,
) -> tuple[list[LayerScoreRow], list[LayerSummaryRow]]:
    """Score every prompt at every probe layer without invoking evolution."""

    if not prompts:
        raise ValueError("diagnostic prompt batch cannot be empty")
    if extractor is None:
        def model_extractor(base: str, prompt: str) -> DeltaExample:
            return extract_appended_delta(model, base, prompt, pooling=pooling)

        extractor = model_extractor
    rows: list[LayerScoreRow] = []
    for example in prompts:
        delta = extractor(baseline, example.prompt)
        aggregate, layer_scores = probe.score(delta)
        aggregate_score = float(aggregate.reshape(-1)[0])
        for layer in probe.layer_indices:
            score = float(layer_scores[layer].reshape(-1)[0])
            rows.append(
                LayerScoreRow(
                    example_id=example.example_id,
                    payload_id=example.payload_id,
                    source_dataset=example.source_dataset,
                    semantic_category=example.semantic_category,
                    attack_family=example.attack_family,
                    layer=layer,
                    drift_score=score,
                    predicted_unsafe=int(score >= threshold),
                    delta_l2_norm=float(torch.linalg.vector_norm(delta[layer].float())),
                    aggregate_drift_score=aggregate_score,
                    aggregate_predicted_unsafe=int(aggregate_score >= threshold),
                    prompt=example.prompt,
                )
            )

    grouped: dict[tuple[str, int], list[LayerScoreRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.attack_family, row.layer)].append(row)
    summaries = []
    for (family, layer), values in sorted(grouped.items()):
        scores = [row.drift_score for row in values]
        norms = [row.delta_l2_norm for row in values]
        summaries.append(
            LayerSummaryRow(
                attack_family=family,
                layer=layer,
                count=len(values),
                mean_drift_score=statistics.fmean(scores),
                sample_std_drift_score=_sample_std(scores),
                detection_rate=statistics.fmean(
                    row.predicted_unsafe for row in values
                ),
                mean_delta_l2_norm=statistics.fmean(norms),
                sample_std_delta_l2_norm=_sample_std(norms),
            )
        )
    return rows, summaries


def _write_csv(rows: Sequence[object], path: str | Path) -> Path:
    if not rows:
        raise ValueError("cannot write an empty diagnostic table")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = [asdict(row) for row in rows]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    return destination


def plot_layer_profiles(
    summaries: Sequence[LayerSummaryRow], path: str | Path
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "raw_control": "#577590",
        "translated": "#F4A261",
        "acrostic_encoding": "#D1495B",
    }
    fig, (score_ax, norm_ax) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    families = sorted({row.attack_family for row in summaries})
    for family in families:
        values = sorted(
            (row for row in summaries if row.attack_family == family),
            key=lambda row: row.layer,
        )
        layers = [row.layer for row in values]
        color = colors.get(family)
        score_ax.errorbar(
            layers,
            [row.mean_drift_score for row in values],
            yerr=[row.sample_std_drift_score for row in values],
            marker="o",
            linewidth=2.5,
            capsize=3,
            label=family.replace("_", " "),
            color=color,
        )
        norm_ax.errorbar(
            layers,
            [row.mean_delta_l2_norm for row in values],
            yerr=[row.sample_std_delta_l2_norm for row in values],
            marker="o",
            linewidth=2.5,
            capsize=3,
            label=family.replace("_", " "),
            color=color,
        )
    score_ax.axhline(0.5, color="#6C757D", linestyle="--", label="0.5 threshold")
    score_ax.set(title="Layerwise Probe Alignment", ylabel="Mean drift probability", ylim=(0, 1))
    norm_ax.set(
        title="Raw Activation-Delta Magnitude",
        xlabel="Hooked full-attention layer",
        ylabel="Mean delta L2 norm",
    )
    norm_ax.set_xticks(sorted({row.layer for row in summaries}))
    score_ax.legend(frameon=True, ncol=2)
    norm_ax.legend(frameon=True, ncol=3)
    for axis in (score_ax, norm_ax):
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--probe", required=True, help="saved DriftProbe checkpoint")
    parser.add_argument("--dataset", help="optional CSV with an instruction column")
    parser.add_argument(
        "--split", choices=("heldout", "active", "all"), default="heldout"
    )
    parser.add_argument("--limit", type=int, help="optional payload limit")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("translated", "acrostic_encoding"),
        default=("translated", "acrostic_encoding"),
    )
    parser.add_argument(
        "--raw-control", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--output-dir", default="outputs/layer_diagnostic", help="artifact directory"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    payloads = (
        csv_payloads(args.dataset)
        if args.dataset
        else benchmark_payloads(config, split=args.split)
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        payloads = payloads[: args.limit]
    prompts = build_diagnostic_prompts(
        payloads, families=args.families, raw_control=args.raw_control
    )
    loaded = load_model(config.model)
    probe = DriftProbe.load(args.probe)
    rows, summaries = diagnose_layers(
        loaded.hooks,
        probe,
        prompts,
        baseline=config.evolution.baseline_prompt,
        pooling=config.probe.pooling,
        threshold=config.probe.threshold,
    )
    output_dir = Path(args.output_dir)
    scores_path = _write_csv(rows, output_dir / "per_layer_scores.csv")
    summary_path = _write_csv(summaries, output_dir / "layer_summary.csv")
    chart_path = plot_layer_profiles(summaries, output_dir / "layer_profiles.png")
    metadata = {
        "probe": str(args.probe),
        "payload_count": len(payloads),
        "prompt_count": len(prompts),
        "layers": probe.layer_indices,
        "families": sorted({prompt.attack_family for prompt in prompts}),
        "threshold": config.probe.threshold,
        "scores_csv": str(scores_path),
        "summary_csv": str(summary_path),
        "chart": str(chart_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
