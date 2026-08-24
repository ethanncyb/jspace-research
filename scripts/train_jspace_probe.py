# Train and evaluate JSpace probes on pooled Jacobian-lens features.
#
# Pipeline:
#   1. Load model + Jacobian lens (scripts/_common.py).
#   2. Extract top-k lens readouts for every record (cached on disk, keyed by
#      prompt hash, so pooling/vocab/head iterations never re-run the model).
#   3. Build the concept vocabulary on train traces only (no dev/test leakage).
#   4. For each pooling x head combination: featurize, train, evaluate on
#      dev / final_test / heldout_families.
#   5. Score the exact-keyword baseline on the same traces for comparison.
#   6. Write per-example + summary CSVs and report.md under --output-dir.
#
# Usage:
#   python scripts/build_jspace_dataset.py --smoke
#   python scripts/train_jspace_probe.py --records-dir data/experiments/jspace --smoke
from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch
from _common import load_model_and_lens

from promptguard.config import load_config
from promptguard.drift_probe import binary_auc
from promptguard.eval_harness import EvalRecord, load_records
from promptguard.jspace_features import (
    ConceptClusters,
    JSpaceFeatureExtractor,
    SparseTrace,
    build_concept_vocab,
    featurize,
    keyword_baseline_score,
    seed_token_ids,
)
from promptguard.jspace_probe import JSpaceProbe

PARTITIONS = ("dev", "final_test", "heldout_families")


def parse_layers(spec: str) -> list[int]:
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, stop = part.split("-", 1)
            layers.extend(range(int(start), int(stop) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_partition_records(records_dir: Path) -> dict[str, list[EvalRecord]]:
    records: dict[str, list[EvalRecord]] = {}
    for name in ("train", *PARTITIONS):
        path = records_dir / f"{name}.csv"
        if path.exists():
            records[name] = load_records(path)
    if "train" not in records:
        raise FileNotFoundError(f"{records_dir / 'train.csv'} is required")
    return records


def extract_all(
    extractor: JSpaceFeatureExtractor,
    records: dict[str, list[EvalRecord]],
    *,
    cache_dir: Path,
) -> dict[str, list[SparseTrace]]:
    traces: dict[str, list[SparseTrace]] = {}
    for name, partition in records.items():
        prompts = [record.prompt for record in partition]
        print(f"extracting {name}: {len(prompts)} prompts")
        traces[name] = extractor.extract_many(prompts, cache_dir=cache_dir / name)
    return traces


def featurize_many(
    traces: Sequence[SparseTrace],
    concept_vocab: Sequence[int],
    *,
    pooling: str,
    clusters: ConceptClusters | None = None,
) -> list[dict[int, torch.Tensor]]:
    examples: list[dict[int, torch.Tensor]] = []
    for trace in traces:
        features = featurize(trace, concept_vocab, pooling=pooling)
        if clusters is not None:
            cluster_features = clusters.aggregate(features)
            features = {
                layer: torch.cat([features[layer], cluster_features[layer]])
                for layer in features
            }
        examples.append(features)
    return examples


def per_example_rows(
    probe: JSpaceProbe,
    examples: Sequence[dict[int, torch.Tensor]],
    records: Sequence[EvalRecord],
    *,
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (example, record) in enumerate(zip(examples, records, strict=True)):
        aggregate, layer_values = probe.score(example)
        score = float(aggregate.reshape(-1)[0])
        row: dict[str, object] = {
            "example_id": index,
            "category": record.category,
            "label": record.label,
            "jspace_score": score,
            "predicted_unsafe": int(score >= threshold),
            "prompt": record.prompt,
        }
        row.update(
            {
                f"layer_{layer}_score": float(value.reshape(-1)[0])
                for layer, value in layer_values.items()
            }
        )
        rows.append(row)
    return rows


def write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def keyword_baseline(
    traces: Sequence[SparseTrace],
    records: Sequence[EvalRecord],
    seed_ids: Sequence[int],
) -> float:
    labels = [record.label for record in records]
    scores = [keyword_baseline_score(trace, seed_ids) for trace in traces]
    return binary_auc(labels, scores)


def hard_stop_demo(
    hf_model,
    tokenizer,
    probe: JSpaceProbe,
    examples: Sequence[dict[int, torch.Tensor]],
    records: Sequence[EvalRecord],
    *,
    threshold: float,
    refusal: str,
    max_new_tokens: int = 32,
) -> list[dict[str, object]]:
    """Generation-time wiring: score, then refuse or generate (hard stop)."""

    rows: list[dict[str, object]] = []
    for example, record in zip(examples, records, strict=True):
        score = float(probe.score(example)[0].reshape(-1)[0])
        triggered = score >= threshold
        if triggered:
            text = refusal
        else:
            ids = tokenizer(record.prompt, return_tensors="pt").to(hf_model.device)
            output = hf_model.generate(**ids, max_new_tokens=max_new_tokens)
            text = tokenizer.decode(output[0][ids.input_ids.shape[1] :])
        rows.append(
            {
                "category": record.category,
                "label": record.label,
                "jspace_score": score,
                "intervention_triggered": int(triggered),
                "generated_text": text,
                "prompt": record.prompt,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--records-dir", default="data/experiments/jspace")
    parser.add_argument("--output-dir", default="outputs/jspace_probe")
    parser.add_argument("--cache-dir", default="outputs/jspace_cache")
    parser.add_argument("--layers", default="21-30")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-concepts", type=int, default=8192)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--poolings", default="max,topm_mass,mean,last_token")
    parser.add_argument("--heads", default="linear,mlp")
    parser.add_argument("--with-clusters", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=64)
    parser.add_argument("--generate", action="store_true", help="hard-stop demo on dev")
    parser.add_argument("--limit-per-file", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="30 records per partition, linear head only, 5 epochs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    records_dir = Path(args.records_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    poolings = [p.strip() for p in args.poolings.split(",") if p.strip()]
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    limit = args.limit_per_file or (30 if args.smoke else 0)
    epochs = 5 if args.smoke else config.probe.epochs

    records = load_partition_records(records_dir)
    if limit:
        records = {name: r[:limit] for name, r in records.items()}

    hf_model, model, lens, tokenizer, device = load_model_and_lens()
    layers = parse_layers(args.layers)
    extractor = JSpaceFeatureExtractor(
        model, lens, layers, top_k=args.top_k, max_seq_len=config.model.max_length
    )
    traces = extract_all(extractor, records, cache_dir=cache_dir)

    concept_vocab = build_concept_vocab(
        traces["train"], max_concepts=args.max_concepts, min_df=args.min_df
    )
    print(f"concept vocabulary: {len(concept_vocab)} tokens")
    seed_ids = seed_token_ids(tokenizer)

    clusters = None
    if args.with_clusters:
        clusters = ConceptClusters.from_unembedding(
            model, concept_vocab, n_clusters=args.n_clusters, seed=args.seed
        )
    feature_dim = len(concept_vocab) + (clusters.n_clusters if clusters else 0)

    labels = {
        name: [record.label for record in partition]
        for name, partition in records.items()
    }
    summary_rows: list[dict[str, object]] = []
    best: tuple[float, str, str] | None = None  # (dev_auc, pooling, head)

    for pooling in poolings:
        featurized = {
            name: featurize_many(
                partition, concept_vocab, pooling=pooling, clusters=clusters
            )
            for name, partition in traces.items()
        }
        for head in heads:
            probe = JSpaceProbe(
                layers,
                feature_dim,
                aggregate=config.probe.aggregate,
                head=head,
            )
            losses = probe.fit(
                featurized["train"],
                labels["train"],
                epochs=epochs,
                learning_rate=config.probe.learning_rate,
                batch_size=config.probe.batch_size,
                weight_decay=config.probe.weight_decay,
                layer_loss_weight=config.probe.layer_loss_weight,
                seed=args.seed,
            )
            row: dict[str, object] = {
                "pooling": pooling,
                "head": head,
                "final_train_loss": losses[-1],
                "train_auc": probe.evaluate(
                    featurized["train"], labels["train"]
                ).auc,
            }
            for name in PARTITIONS:
                if name not in featurized:
                    continue
                metrics = probe.evaluate(
                    featurized[name],
                    labels[name],
                    threshold=config.probe.threshold,
                )
                row[f"{name}_auc"] = metrics.auc
                row[f"{name}_unsafe_recall"] = metrics.unsafe_recall
                row[f"{name}_benign_pass_rate"] = metrics.benign_pass_rate
                rows = per_example_rows(
                    probe,
                    featurized[name],
                    records[name],
                    threshold=config.probe.threshold,
                )
                write_csv(
                    rows, output_dir / f"examples_{pooling}_{head}_{name}.csv"
                )
            summary_rows.append(row)
            dev_auc = float(row.get("dev_auc", math.nan))
            if not math.isnan(dev_auc) and (best is None or dev_auc > best[0]):
                best = (dev_auc, pooling, head)
                probe.save(
                    output_dir / "best_probe.pt",
                    pooling=pooling,
                    head=head,
                    concept_vocab=concept_vocab,
                    layers=layers,
                )
            print(json.dumps({k: v for k, v in row.items()}, default=str))

    baseline_row: dict[str, object] = {"pooling": "keyword_baseline", "head": "-"}
    for name in PARTITIONS:
        if name in traces:
            baseline_row[f"{name}_auc"] = keyword_baseline(
                traces[name], records[name], seed_ids
            )
    summary_rows.append(baseline_row)

    write_csv(summary_rows, output_dir / "summary.csv")

    lines = [
        "# JSpace probe report",
        "",
        f"model layers: {layers}, top_k={args.top_k}, "
        f"concepts={len(concept_vocab)}, clusters={args.with_clusters}",
        f"train size: {len(records['train'])}",
        "",
        "| pooling | head | dev AUC | final AUC | heldout AUC |",
        "|---|---|---|---|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {pooling} | {head} | {dev} | {final} | {heldout} |".format(
                pooling=row["pooling"],
                head=row["head"],
                dev=_fmt(row.get("dev_auc")),
                final=_fmt(row.get("final_test_auc")),
                heldout=_fmt(row.get("heldout_families_auc")),
            )
        )
    lines.append("")
    if best is not None:
        lines.append(
            f"best on dev: pooling={best[1]}, head={best[2]} (AUC {best[0]:.3f}); "
            f"checkpoint: {output_dir / 'best_probe.pt'}"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_dir / 'summary.csv'} and {output_dir / 'report.md'}")

    if args.generate and best is not None:
        _, pooling, head = best
        probe = JSpaceProbe.load(output_dir / "best_probe.pt")
        demo_records = records["dev"][:10]
        demo_traces = traces["dev"][:10]
        demo_examples = featurize_many(
            demo_traces, concept_vocab, pooling=pooling, clusters=clusters
        )
        rows = hard_stop_demo(
            hf_model,
            tokenizer,
            probe,
            demo_examples,
            demo_records,
            threshold=config.intervention.threshold,
            refusal=config.intervention.refusal,
        )
        write_csv(rows, output_dir / "hard_stop_demo.csv")
        print(f"wrote {output_dir / 'hard_stop_demo.csv'}")


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


if __name__ == "__main__":
    main()
