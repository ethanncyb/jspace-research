"""Batch evaluation for activation capture, probe scoring, and intervention."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from promptguard.config import ResearchConfig, load_config
from promptguard.drift_probe import DriftProbe, binary_auc, extract_appended_delta
from promptguard.evolving_attacker import heldout_translation_wrapper
from promptguard.intervention import InterventionController
from promptguard.model_hooks import HookedModel, load_model


@dataclass(frozen=True)
class EvalRecord:
    baseline: str
    prompt: str
    label: int
    category: str


def load_records(path: str | Path) -> list[EvalRecord]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"baseline", "prompt", "label", "category"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"evaluation CSV requires columns {sorted(required)}")
        return [
            EvalRecord(
                row["baseline"], row["prompt"], int(row["label"]), row["category"]
            )
            for row in reader
        ]


def config_records(config: ResearchConfig) -> list[EvalRecord]:
    baseline = config.evolution.baseline_prompt
    return [
        *[
            EvalRecord(baseline, prompt, 0, "benign")
            for prompt in config.data.benign_prompts
        ],
        *[
            EvalRecord(baseline, prompt, 1, "injected_or_jailbreak")
            for prompt in config.data.malicious_prompts
        ],
        *[
            EvalRecord(
                baseline,
                heldout_translation_wrapper(prompt),
                1,
                "heldout_translation_family",
            )
            for prompt in config.data.heldout_malicious_prompts
        ],
    ]


def evaluate(
    model: HookedModel,
    probe: DriftProbe,
    config: ResearchConfig,
    records: Sequence[EvalRecord],
    *,
    output_csv: str | Path,
    summary_csv: str | Path,
    generate: bool = False,
    max_new_tokens: int = 32,
) -> dict[str, float | int]:
    controller = InterventionController(probe, config.intervention)
    rows: list[dict[str, object]] = []
    scores: list[float] = []
    labels: list[int] = []
    threshold = config.probe.threshold

    for index, record in enumerate(records):
        delta = extract_appended_delta(
            model,
            record.baseline,
            record.prompt,
            pooling=config.probe.pooling,
        )
        aggregate, layer_values = probe.score(delta)
        score = float(aggregate.reshape(-1)[0])
        triggered = score >= config.intervention.threshold
        generated_text = ""
        applied_mode = config.intervention.mode if triggered else "none"
        if generate:
            result = controller.run(
                model,
                baseline_text=record.baseline,
                prompt=record.prompt,
                pooling=config.probe.pooling,
                max_new_tokens=max_new_tokens,
            )
            generated_text = result.text
            triggered = result.triggered
            applied_mode = result.mode
        row: dict[str, object] = {
            "example_id": index,
            "category": record.category,
            "label": record.label,
            "drift_score": score,
            "predicted_unsafe": int(score >= threshold),
            "intervention_triggered": int(triggered),
            "intervention_mode": applied_mode,
            "generated_text": generated_text,
            "prompt": record.prompt,
        }
        row.update(
            {
                f"layer_{layer}_score": float(value.reshape(-1)[0])
                for layer, value in layer_values.items()
            }
        )
        rows.append(row)
        scores.append(score)
        labels.append(record.label)

    predictions = [int(score >= threshold) for score in scores]
    tp = sum(p == y == 1 for p, y in zip(predictions, labels, strict=True))
    tn = sum(p == y == 0 for p, y in zip(predictions, labels, strict=True))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels, strict=True))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels, strict=True))
    summary: dict[str, float | int] = {
        "count": len(rows),
        "auc": binary_auc(labels, scores),
        "accuracy": (tp + tn) / max(len(rows), 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
        "unsafe_recall": tp / max(tp + fn, 1),
        "benign_pass_rate": tn / max(tn + fp, 1),
        "intervention_trigger_rate": sum(
            int(row["intervention_triggered"]) for row in rows
        )
        / max(len(rows), 1),
    }
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary_destination = Path(summary_csv)
    summary_destination.parent.mkdir(parents=True, exist_ok=True)
    with summary_destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--probe", required=True)
    parser.add_argument("--dataset", help="optional CSV; otherwise config.data is used")
    parser.add_argument("--output", default="outputs/eval/examples.csv")
    parser.add_argument("--summary", default="outputs/eval/summary.csv")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    loaded = load_model(config.model)
    probe = DriftProbe.load(args.probe)
    records = load_records(args.dataset) if args.dataset else config_records(config)
    summary = evaluate(
        loaded.hooks,
        probe,
        config,
        records,
        output_csv=args.output,
        summary_csv=args.summary,
        generate=args.generate,
        max_new_tokens=args.max_new_tokens,
    )
    print(summary)


if __name__ == "__main__":
    main()
