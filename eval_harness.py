"""Command-line collection, training, evaluation, and guarded generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from continual_update import ContinualUpdateStore, FlaggedDeltaRecord
from intervention import InterventionConfig, guarded_generate
from model_hooks import (
    ResidualHookController,
    hidden_size,
    load_model_and_tokenizer,
    resolve_residual_stack,
    run_paired_prefill,
    validate_layer_layout,
)
from probe import (
    DatasetRow,
    LayerProbeDetector,
    classification_metrics,
    load_dataset,
    train_probes,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration must be a YAML mapping")
    return value


def _model(config: Mapping[str, Any]):
    return load_model_and_tokenizer(
        config["model_id"],
        device=config.get("device", "auto"),
        dtype=config.get("dtype", "auto"),
    )[:2]


def _artifact_rows(payload: Mapping[str, Any]) -> list[DatasetRow]:
    return [DatasetRow(**row) for row in payload["rows"]]


def collect(
    config: Mapping[str, Any], dataset_path: str | Path, output: str | Path
) -> Path:
    model, tokenizer = _model(config)
    layers = validate_layer_layout(model, config["full_attention_indices"])
    rows = load_dataset(
        dataset_path, field_map=config.get("field_map"), tokenizer=tokenizer
    )
    features: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for row in rows:
        controller = ResidualHookController(resolve_residual_stack(model), layers)
        with controller:
            run_paired_prefill(
                model,
                tokenizer,
                row.clean_prompt,
                row.appended_text,
                controller,
                appended_token_ids=row.metadata.get("_candidate_suffix_token_ids"),
                max_length=config.get("maximum_length", 2048),
            )
        for layer in layers:
            features[layer].append(controller.deltas[layer].squeeze(0))
    payload = {
        "format_version": 1,
        "model_id": config["model_id"],
        "layers": layers,
        "hidden_dim": hidden_size(model),
        "rows": [asdict(row) for row in rows],
        "features": {layer: torch.stack(values) for layer, values in features.items()},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def train(
    config: Mapping[str, Any], features_path: str | Path, output: str | Path
) -> dict[str, Any]:
    payload = torch.load(features_path, map_location="cpu", weights_only=False)
    rows = _artifact_rows(payload)
    settings = config.get("probe_training", {})
    result = train_probes(
        payload["features"],
        [row.label for row in rows],
        rows,
        model_id=payload["model_id"],
        aggregation=config.get("aggregation", "max"),
        validation_fraction=settings.get("validation_fraction", 0.2),
        epochs=settings.get("epochs", 100),
        learning_rate=settings.get("learning_rate", 0.01),
        seed=config.get("seed", 0),
    )
    result.detector.save(output)
    return result.validation_metrics


def _confusion(label: int, prediction: int) -> str:
    return ("true_" if label == prediction else "false_") + (
        "positive" if prediction else "negative"
    )


def evaluate(
    config: Mapping[str, Any],
    features_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    exercise_intervention: bool = False,
) -> dict[str, Any]:
    payload = torch.load(features_path, map_location="cpu", weights_only=False)
    rows = _artifact_rows(payload)
    detector = LayerProbeDetector.load(
        checkpoint,
        expected_layers=payload["layers"],
        expected_hidden_dim=payload["hidden_dim"],
        expected_model_id=payload["model_id"],
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_layer: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []
    scores: list[float] = []
    labels: list[int] = []
    guarded_results = None
    if exercise_intervention:
        guard_model, guard_tokenizer = _model(config)
        validate_layer_layout(guard_model, detector.layers)
        settings = config.get("intervention", {})
        guard_config = InterventionConfig(
            mode=settings.get("mode", "circuit_breaker"),
            threshold=settings.get("threshold"),
            beta=settings.get("beta", 1.0),
            refusal_text=settings.get("refusal_text", "I cannot process that request."),
        )
        guarded_results = [
            guarded_generate(
                guard_model,
                guard_tokenizer,
                row.clean_prompt,
                row.appended_text,
                detector,
                guard_config,
                max_length=config.get("maximum_length", 2048),
                generation_kwargs={"do_sample": False, "max_new_tokens": 1},
            )
            for row in rows
        ]
        detector.cpu()
    for index, row in enumerate(rows):
        guarded = guarded_results[index] if guarded_results is not None else None
        guarded_trigger_layers = (
            {trigger.layer for trigger in guarded.triggers} if guarded else set()
        )
        probabilities: dict[int, float] = {}
        distances: dict[int, float] = {}
        trigger_layers: list[int] = []
        for layer in detector.layers:
            delta = payload["features"][layer][index]
            probability = float(detector.probability(layer, delta).item())
            distance = float(detector.benign_distance(layer, delta).item())
            threshold = detector.threshold(layer)
            probabilities[layer] = probability
            distances[layer] = distance
            if probability > threshold:
                trigger_layers.append(layer)
            per_layer.append(
                {
                    "sample": row.id,
                    "label": row.label,
                    "layer": layer,
                    "cosine_distance": distance,
                    "l2_distance": float(torch.linalg.vector_norm(delta).item()),
                    "probe_probability": probability,
                    "threshold": threshold,
                    "triggered": probability > threshold,
                    "intervention": (
                        guarded.layer_diagnostics.get(layer, {}).get(
                            "intervention",
                            "hard_stop" if layer in guarded_trigger_layers else "none",
                        )
                        if guarded
                        else "not_exercised"
                    ),
                }
            )
        score = detector.aggregate(probabilities)
        prediction = int(score >= 0.5)
        scores.append(score)
        labels.append(row.label)
        per_example.append(
            {
                "sample": row.id,
                "label": row.label,
                "aggregate_probe_probability": score,
                "aggregate_cosine_distance": detector.aggregate(distances),
                "prediction": prediction,
                "confusion_category": _confusion(row.label, prediction),
                "first_trigger_layer": min(trigger_layers) if trigger_layers else "",
                "guarded_outcome": (
                    "hard_stop"
                    if guarded and guarded.stopped
                    else "intervened"
                    if guarded and guarded.triggers
                    else "allowed"
                    if guarded
                    else "not_exercised"
                ),
            }
        )
    metrics = classification_metrics(labels, scores)
    summary = {
        **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
        "true_negative": metrics["confusion_matrix"][0][0],
        "false_positive": metrics["confusion_matrix"][0][1],
        "false_negative": metrics["confusion_matrix"][1][0],
        "true_positive": metrics["confusion_matrix"][1][1],
        "model_id": payload["model_id"],
        "layers": ",".join(map(str, detector.layers)),
        "aggregation": detector.aggregation,
    }
    _write_csv(output_dir / "per_layer.csv", per_layer)
    _write_csv(output_dir / "per_example.csv", per_example)
    _write_csv(output_dir / "summary.csv", [summary])
    continual = config.get("continual_learning", {})
    auto_path = output_dir / continual.get("auto_buffer", "auto_buffer.jsonl")
    review_path = output_dir / continual.get("manual_review", "manual_review.csv")
    store = ContinualUpdateStore(
        auto_path,
        review_path,
        strict_confidence=continual.get("strict_confidence", 0.995),
    )
    for index, example in enumerate(per_example):
        if example["prediction"]:
            first = example["first_trigger_layer"]
            layer = int(first) if first != "" else detector.layers[0]
            store.route(
                FlaggedDeltaRecord(
                    rows[index].id,
                    rows[index].clean_prompt + rows[index].appended_text,
                    float(example["aggregate_probe_probability"]),
                    layer,
                    str(checkpoint),
                    payload["features"][layer][index],
                    rows[index].metadata,
                )
            )
    if not auto_path.exists():
        auto_path.touch()
    if not review_path.exists():
        _write_csv(review_path, [], fieldnames=ContinualUpdateStore.REVIEW_FIELDS)
    return summary


def _write_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    *,
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        if names:
            writer.writeheader()
            writer.writerows(rows)


def generate(
    config: Mapping[str, Any],
    checkpoint: str | Path,
    clean_prompt: str,
    appended_text: str,
) -> dict[str, Any]:
    model, tokenizer = _model(config)
    layers = validate_layer_layout(model, config["full_attention_indices"])
    detector = LayerProbeDetector.load(
        checkpoint,
        expected_layers=layers,
        expected_hidden_dim=hidden_size(model),
        expected_model_id=config["model_id"],
    )
    intervention = config.get("intervention", {})
    result = guarded_generate(
        model,
        tokenizer,
        clean_prompt,
        appended_text,
        detector,
        InterventionConfig(
            mode=intervention.get("mode", "circuit_breaker"),
            threshold=intervention.get("threshold"),
            beta=intervention.get("beta", 1.0),
            refusal_text=intervention.get(
                "refusal_text", "I cannot process that request."
            ),
        ),
        max_length=config.get("maximum_length", 2048),
        generation_kwargs=config.get("generation", {}),
    )
    return {
        "text": result.text,
        "stopped": result.stopped,
        "triggers": [asdict(trigger) for trigger in result.triggers],
        "layer_diagnostics": result.layer_diagnostics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    subcommands = parser.add_subparsers(dest="command", required=True)
    collect_parser = subcommands.add_parser("collect")
    collect_parser.add_argument("dataset")
    collect_parser.add_argument("--output", required=True)
    train_parser = subcommands.add_parser("train")
    train_parser.add_argument("features")
    train_parser.add_argument("--output", required=True)
    evaluate_parser = subcommands.add_parser("evaluate")
    evaluate_parser.add_argument("features")
    evaluate_parser.add_argument("checkpoint")
    evaluate_parser.add_argument("--output-dir", required=True)
    evaluate_parser.add_argument("--exercise-intervention", action="store_true")
    generate_parser = subcommands.add_parser("generate")
    generate_parser.add_argument("checkpoint")
    generate_parser.add_argument("clean_prompt")
    generate_parser.add_argument("appended_text")
    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("dataset")
    run_parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "collect":
        print(collect(config, args.dataset, args.output))
    elif args.command == "train":
        print(json.dumps(train(config, args.features, args.output), indent=2))
    elif args.command == "evaluate":
        print(
            json.dumps(
                evaluate(
                    config,
                    args.features,
                    args.checkpoint,
                    args.output_dir,
                    exercise_intervention=args.exercise_intervention,
                ),
                indent=2,
            )
        )
    elif args.command == "generate":
        print(
            json.dumps(
                generate(
                    config, args.checkpoint, args.clean_prompt, args.appended_text
                ),
                indent=2,
            )
        )
    else:
        output_dir = Path(args.output_dir)
        features = collect(config, args.dataset, output_dir / "features.pt")
        checkpoint = output_dir / "probe.pt"
        train(config, features, checkpoint)
        print(json.dumps(evaluate(config, features, checkpoint, output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
