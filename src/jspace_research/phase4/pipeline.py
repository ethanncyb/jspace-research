from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from ..model import HuggingFaceModelAdapter, load_tokenizer
from ..phase1.adapters import JacobianLensAdapter, validate_lens_for_layers
from ..phase1.artifacts import load_phase1_handoff
from ..phase1.data import verify_bipia_revision
from ..phase1.jspace import build_normalized_dictionary
from ..phase2.scoring import (
    JUDGE_GATEWAY,
    JUDGE_RUBRIC_SHA256,
    OpenRouterAttackJudge,
)
from ..runtime import (
    append_jsonl,
    atomic_save_figure,
    atomic_write_csv,
    atomic_write_parquet,
    cuda_metadata,
    package_versions,
    read_json,
    read_resumable_jsonl,
    sha256_file,
    update_provenance,
    validate_identity_fields,
)
from . import agentdojo, bipia, injecagent
from .common import completed_records, content_hash, verify_checkout
from .config import Phase4Config
from .detectors import FrozenDetectors

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PACKAGES = (
    "agentdojo",
    "jspace-research",
    "jlens",
    "nltk",
    "openai",
    "pandas",
    "scikit-learn",
    "torch",
    "transformers",
)


def _handoff(config: Phase4Config) -> tuple[Any, FrozenDetectors]:
    handoff = load_phase1_handoff(config.phase1_selected_path, config.phase1)
    detectors = FrozenDetectors.load(config.phase3_dir, handoff.metadata)
    return handoff, detectors


def _identity(config: Phase4Config, handoff: Any, detectors: FrozenDetectors) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase4_config_sha256": config.identity_hash(),
        "phase1_run_id": handoff.metadata["run_id"],
        "phase3_config_sha256": detectors.mean["phase3_config_sha256"],
        "mean_detector_sha256": sha256_file(config.phase3_dir / "mean_detector.pt"),
        "logistic_detector_sha256": sha256_file(
            config.phase3_dir / "logistic_detector.pt"
        ),
    }


def _base_provenance(
    config: Phase4Config, handoff: Any, detectors: FrozenDetectors
) -> dict[str, Any]:
    phase1 = config.phase1.scientific_dict()
    return {
        "schema_version": 1,
        "phase": 4,
        "run_id": f"phase4-{config.identity_hash()[:12]}-{handoff.metadata['run_id']}",
        **_identity(config, handoff, detectors),
        "model": phase1["model"],
        "lens": phase1["lens"],
        "selected_layer": handoff.selected_layer,
        "decomposition": handoff.metadata["decomposition"],
        "detectors": {
            "mean_sha256": sha256_file(config.phase3_dir / "mean_detector.pt"),
            "logistic_sha256": sha256_file(config.phase3_dir / "logistic_detector.pt"),
            "mean_threshold": float(detectors.mean["threshold"]),
            "logistic_threshold": float(detectors.logistic["threshold"]),
            "logistic_feature_count": int(detectors.logistic["feature_token_ids"].numel()),
        },
        "resolved_config": config.scientific_dict(),
        "judge": {
            "gateway": JUDGE_GATEWAY,
            "requested_model": config.judge_model,
            "rubric_sha256": JUDGE_RUBRIC_SHA256,
        },
    }


def _provenance(
    config: Phase4Config,
    handoff: Any,
    detectors: FrozenDetectors,
    updates: dict[str, Any] | None = None,
) -> None:
    update_provenance(
        config.output_dir / "provenance.json",
        _base_provenance(config, handoff, detectors),
        defaults={"generation_gpu": None, "generation_packages": None, "analysis_packages": None},
        updates=updates,
    )


def generate(config: Phase4Config) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("The Phase 4 generate stage requires a CUDA GPU")
    config.validate(require_roots=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    verify_bipia_revision(config.phase1)
    verify_checkout(config.agentdojo_root, config.agentdojo_revision, "AgentDojo")
    verify_checkout(config.injecagent_root, config.injecagent_revision, "InjecAgent")
    handoff, detectors = _handoff(config)
    identity = _identity(config, handoff, detectors)
    _provenance(config, handoff, detectors)

    cached = {
        name: completed_records(config.output_dir / f"{name}_records.jsonl", identity)
        for name in ("bipia", "agentdojo", "injecagent")
    }
    tokenizer = load_tokenizer(config.phase1)
    model = HuggingFaceModelAdapter.load(config.phase1, tokenizer)
    lens = JacobianLensAdapter.load(config.phase1)
    layer = handoff.selected_layer
    validate_lens_for_layers(lens, model.hidden_width, [layer])
    dictionary = build_normalized_dictionary(
        jacobian=lens.jacobian(layer),
        unembedding=model.unembedding(),
        layer=layer,
        device=model.input_device,
        chunk_size=config.phase1.dictionary_chunk_size,
    )
    detectors = detectors.with_dictionary(dictionary)

    bipia.generate(config, model, detectors, cached["bipia"], identity)
    agentdojo.generate(config, model, detectors, cached["agentdojo"], identity)
    injecagent.generate(config, model, detectors, cached["injecagent"], identity)

    _provenance(
        config,
        handoff,
        detectors,
        updates={
            "generation_gpu": cuda_metadata(model_input_device=str(model.input_device)),
            "generation_packages": package_versions(PACKAGES),
            "generation_record_counts": {
                name: len(read_resumable_jsonl(config.output_dir / f"{name}_records.jsonl"))
                for name in ("bipia", "agentdojo", "injecagent")
            },
            "generation_complete": True,
        },
    )
    del dictionary, detectors, lens, model, tokenizer, handoff
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Phase 4 generation complete: {config.output_dir}")
    return config.output_dir


def _judge_bipia(
    config: Phase4Config, records: list[dict[str, Any]], judge: Any | None
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    path = config.output_dir / "bipia_judgments.jsonl"
    expected = {
        row["case_id"]: {
            "schema_version": 1,
            "case_id": row["case_id"],
            "phase4_config_sha256": config.identity_hash(),
            "case_hash": row["case_hash"],
            "judge_gateway": JUDGE_GATEWAY,
            "judge_model": config.judge_model,
            "judge_rubric_sha256": JUDGE_RUBRIC_SHA256,
            "generation_hash": content_hash(row["generated_response"]),
        }
        for row in records
        if row["condition"] == "attack"
    }
    cached: dict[str, dict[str, Any]] = {}
    for value in read_resumable_jsonl(path):
        case_id = value.get("case_id")
        if case_id not in expected or case_id in cached:
            raise RuntimeError(f"Unexpected or duplicate BIPIA judgment at {path}")
        validate_identity_fields(path, value, expected[case_id])
        if value.get("judge_label") not in {"YES", "NO", "UNKNOWN"}:
            raise RuntimeError(f"Incomplete BIPIA judgment at {path}")
        cached[case_id] = value

    active = judge
    by_id = {row["case_id"]: row for row in records}
    for case_id, fields in tqdm(expected.items(), desc="Phase 4 BIPIA judgments"):
        if case_id in cached:
            continue
        if active is None:
            active = OpenRouterAttackJudge(config.judge_model)
        row = by_id[case_id]
        result = active.judge(row["attack_text"], row["generated_response"])
        value = {
            **fields,
            "judge_label": result.label,
            "attack_success": result.label == "YES",
            "returned_model": result.returned_model,
            "provider": result.provider,
            "provider_model": result.provider_model,
        }
        append_jsonl(path, value)
        cached[case_id] = value
    routing = {
        name: sorted({str(row[field]) for row in cached.values() if row.get(field)})
        for name, field in (
            ("returned_models", "returned_model"),
            ("providers", "provider"),
            ("provider_models", "provider_model"),
        )
    }
    return cached, routing


def _detector_rows(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    scope: str,
    subgroup: str | None,
    metric: str,
    detector: str,
    value: float,
    n: int,
    threshold: float | None,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "scope": scope,
        "subgroup": subgroup,
        "detector": detector,
        "metric": metric,
        "value": value,
        "n": n,
        "threshold": threshold,
    }


def _metrics(predictions: pd.DataFrame, detectors: FrozenDetectors) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    thresholds = {
        "mean": float(detectors.mean["threshold"]),
        "logistic": float(detectors.logistic["threshold"]),
    }
    bipia_rows = predictions[predictions.benchmark == "bipia"]
    labels = (bipia_rows.condition == "attack").to_numpy(dtype=np.int64)
    for detector in ("mean", "logistic"):
        scores = bipia_rows[f"{detector}_score"].to_numpy(dtype=float)
        decisions = scores >= thresholds[detector]
        tpr = float(decisions[labels == 1].mean())
        fpr = float(decisions[labels == 0].mean())
        values = {
            "auprc": average_precision_score(labels, scores),
            "auroc": roc_auc_score(labels, scores),
            "tpr": tpr,
            "fpr": fpr,
            "balanced_accuracy": (tpr + 1 - fpr) / 2,
        }
        for metric, value in values.items():
            rows.append(_detector_rows(bipia_rows, benchmark="bipia", scope="overall", subgroup=None, metric=metric, detector=detector, value=float(value), n=len(bipia_rows), threshold=thresholds[detector]))

    dojo = predictions[predictions.benchmark == "agentdojo"]
    for subgroup in [None, *sorted(dojo.subgroup.dropna().unique())]:
        subset = dojo if subgroup is None else dojo[dojo.subgroup == subgroup]
        scope = "overall" if subgroup is None else "suite"
        for detector in ("mean", "logistic"):
            clean = subset[(subset.condition == "control") & subset[f"{detector}_score"].notna()]
            attacked = subset[(subset.condition == "attack") & (subset.injection_exposed == True) & subset[f"{detector}_score"].notna()]  # noqa: E712
            for metric, values in (("fpr", clean[f"{detector}_prediction"]), ("tpr", attacked[f"{detector}_prediction"])):
                rows.append(_detector_rows(subset, benchmark="agentdojo", scope=scope, subgroup=subgroup, metric=metric, detector=detector, value=float(values.mean()), n=len(values), threshold=thresholds[detector]))
        for metric, values in (
            ("clean_utility", subset[subset.condition == "control"].native_utility),
            ("utility_under_attack", subset[subset.condition == "attack"].native_utility),
            ("targeted_asr", subset[subset.condition == "attack"].native_attack_success),
        ):
            rows.append(_detector_rows(subset, benchmark="agentdojo", scope=scope, subgroup=subgroup, metric=metric, detector="native", value=float(values.mean()), n=len(values), threshold=None))

    injec = predictions[predictions.benchmark == "injecagent"]
    for subgroup in [None, *sorted(injec.subgroup.dropna().unique())]:
        subset = injec if subgroup is None else injec[injec.subgroup == subgroup]
        scope = "overall" if subgroup is None else "subgroup"
        for detector in ("mean", "logistic"):
            scores = subset[f"{detector}_score"].astype(float)
            values = {
                "tpr": float(subset[f"{detector}_prediction"].mean()),
                "score_mean": float(scores.mean()),
                "score_median": float(scores.median()),
                "score_q25": float(scores.quantile(0.25)),
                "score_q75": float(scores.quantile(0.75)),
            }
            for metric, value in values.items():
                rows.append(_detector_rows(subset, benchmark="injecagent", scope=scope, subgroup=subgroup, metric=metric, detector=detector, value=value, n=len(subset), threshold=thresholds[detector]))
        valid = subset.native_valid.astype(bool)
        success = subset.native_attack_success.astype(bool)
        for metric, value, n in (
            ("valid_rate", float(valid.mean()), len(valid)),
            ("asr_valid", float(success[valid].mean()) if bool(valid.any()) else float("nan"), int(valid.sum())),
            ("asr_all", float(success.mean()), len(success)),
        ):
            rows.append(_detector_rows(subset, benchmark="injecagent", scope=scope, subgroup=subgroup, metric=metric, detector="native", value=value, n=n, threshold=None))
    return pd.DataFrame(rows)


def _plot(config: Phase4Config, metrics: pd.DataFrame) -> Path:
    tpr = metrics[(metrics.metric == "tpr") & (metrics.scope == "overall")]
    benchmarks = [name for name in ("bipia", "agentdojo", "injecagent") if name in set(tpr.benchmark)]
    x = np.arange(len(benchmarks))
    figure, axis = plt.subplots(figsize=(8, 5))
    for offset, detector in ((-0.18, "mean"), (0.18, "logistic")):
        values = [float(tpr[(tpr.benchmark == name) & (tpr.detector == detector)].value.iloc[0]) for name in benchmarks]
        axis.bar(x + offset, values, 0.36, label=detector.capitalize())
    axis.set_xticks(x)
    axis.set_xticklabels([name.upper() if name == "bipia" else name for name in benchmarks])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("TPR at frozen Phase 3 threshold")
    axis.set_title("Frozen J-Space Detector Transfer")
    axis.legend()
    figure.tight_layout()
    path = config.output_dir / "phase4_detector_transfer.png"
    atomic_save_figure(path, figure, dpi=180)
    plt.close(figure)
    return path


def analyze(config: Phase4Config, *, judge: Any | None = None) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    handoff, detectors = _handoff(config)
    identity = _identity(config, handoff, detectors)
    provenance = read_json(config.output_dir / "provenance.json")
    if provenance.get("generation_complete") is not True:
        raise RuntimeError("Phase 4 generation is incomplete; run --stage generate first")
    records = {
        name: list(completed_records(config.output_dir / f"{name}_records.jsonl", identity).values())
        for name in ("bipia", "agentdojo", "injecagent")
    }
    if any(not values for values in records.values()):
        raise RuntimeError("Phase 4 generation is incomplete; run --stage generate first")
    actual_counts = {name: len(values) for name, values in records.items()}
    if provenance.get("generation_record_counts") != actual_counts:
        raise RuntimeError("Phase 4 generation record counts do not match provenance")
    judgments, routing = _judge_bipia(config, records["bipia"], judge)

    rows: list[dict[str, Any]] = []
    for benchmark, values in records.items():
        for value in values:
            judgment = judgments.get(value["case_id"])
            rows.append(
                {
                    "case_id": value["case_id"],
                    "benchmark": benchmark,
                    "task": value.get("task"),
                    "subgroup": value.get("subgroup"),
                    "condition": value["condition"],
                    "injection_exposed": value.get("injection_exposed"),
                    "mean_score": value.get("mean_score"),
                    "mean_prediction": value.get("mean_prediction"),
                    "logistic_score": value.get("logistic_score"),
                    "logistic_prediction": value.get("logistic_prediction"),
                    "generated_response": value.get("generated_response"),
                    "native_valid": value.get("native_valid"),
                    "native_utility": value.get("native_utility"),
                    "native_attack_success": value.get("native_attack_success"),
                    "judge_label": judgment.get("judge_label") if judgment else None,
                    "attack_success": judgment.get("attack_success") if judgment else None,
                }
            )
    predictions = pd.DataFrame(rows).sort_values(["benchmark", "case_id"])
    for column in ("injection_exposed", "mean_prediction", "logistic_prediction", "native_valid", "native_utility", "native_attack_success", "attack_success"):
        predictions[column] = pd.array(predictions[column], dtype="boolean")
    predictions_path = config.output_dir / "phase4_predictions.parquet"
    atomic_write_parquet(predictions_path, predictions)
    metrics = _metrics(predictions, detectors)
    metrics_path = config.output_dir / "phase4_metrics.csv"
    atomic_write_csv(metrics_path, metrics)
    plot_path = _plot(config, metrics)
    _provenance(
        config,
        handoff,
        detectors,
        updates={
            "analysis_packages": package_versions(PACKAGES),
            "judge_routing": routing,
            "artifacts": {
                "predictions": {"path": predictions_path.name, "sha256": sha256_file(predictions_path)},
                "metrics": {"path": metrics_path.name, "sha256": sha256_file(metrics_path)},
                "plot": {"path": plot_path.name, "sha256": sha256_file(plot_path)},
            },
            "complete": True,
        },
    )
    print(f"Phase 4 analysis complete: {predictions_path}")
    return predictions_path


def run(config: Phase4Config, stage: str, *, judge: Any | None = None) -> Path:
    if stage == "generate":
        return generate(config)
    if stage == "analyze":
        return analyze(config, judge=judge)
    if stage == "all":
        generate(config)
        return analyze(config, judge=judge)
    raise ValueError(f"Unknown Phase 4 stage: {stage}")
