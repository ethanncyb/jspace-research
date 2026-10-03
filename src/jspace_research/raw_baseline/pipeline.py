from __future__ import annotations

import gc
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from ..model import HuggingFaceModelAdapter, load_tokenizer
from ..phase3.pipeline import compute_metrics, select_threshold
from ..phase4 import agentdojo, bipia, injecagent
from ..phase4.common import completed_records, verify_checkout
from ..phase4.detectors import FrozenDetectors
from ..phase4.pipeline import _metrics as transfer_metrics
from ..runtime import (
    atomic_save_figure,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_parquet,
    cuda_metadata,
    package_versions,
    read_json,
    read_resumable_jsonl,
    sha256_file,
    update_provenance,
)
from .artifacts import RawActivationCache, load_raw_activation_cache, load_raw_detector
from .config import RawBaselineConfig
from .detectors import RawDetectors

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PACKAGES = ("agentdojo", "jspace-research", "pandas", "scikit-learn", "torch", "transformers")
RECORD_NAMES = ("bipia", "agentdojo", "injecagent")


def _reference_provenance(config: RawBaselineConfig) -> dict[str, Any]:
    phase4 = read_json(config.jspace_phase4_dir / "provenance.json")
    required = {
        "phase1_run_id": None,
        "phase4_config_sha256": config.phase4_config(config.jspace_phase4_dir).identity_hash(),
        "generation_complete": True,
        "complete": True,
    }
    for key, expected in required.items():
        if key not in phase4 or (expected is not None and phase4[key] != expected):
            raise RuntimeError("Completed Phase 4 provenance does not match the raw baseline")
    for name in (
        "bipia_test_manifest.jsonl",
        "bipia_records.jsonl",
        "agentdojo_records.jsonl",
        "injecagent_records.jsonl",
        "phase4_predictions.parquet",
        "phase4_metrics.csv",
    ):
        if not (config.jspace_phase4_dir / name).is_file():
            raise RuntimeError(f"Completed Phase 4 artifact is missing: {name}")
    return phase4


def _base_provenance(
    config: RawBaselineConfig, cache: RawActivationCache, phase4: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "same_layer_raw_residual_baseline",
        "run_id": f"raw-{config.identity_hash()[:12]}-{cache.handoff.metadata['run_id']}",
        "raw_baseline_config_sha256": config.identity_hash(),
        "phase1_run_id": cache.handoff.metadata["run_id"],
        "phase1_config_sha256": cache.handoff.metadata["config_sha256"],
        "manifest_sha256": cache.handoff.metadata["manifest_sha256"],
        "selected_layer": cache.handoff.selected_layer,
        "raw_width": cache.width,
        "jspace_phase3_mean_sha256": sha256_file(config.jspace_phase3_dir / "mean_detector.pt"),
        "jspace_phase3_logistic_sha256": sha256_file(
            config.jspace_phase3_dir / "logistic_detector.pt"
        ),
        "jspace_phase4_run_id": phase4["run_id"],
        "jspace_phase4_config_sha256": phase4["phase4_config_sha256"],
        "bipia_test_manifest_sha256": phase4["bipia_test_manifest_sha256"],
        "model": config.phase1.scientific_dict()["model"],
        "resolved_config": config.scientific_dict(),
    }


def _provenance(
    config: RawBaselineConfig,
    cache: RawActivationCache,
    phase4: dict[str, Any],
    updates: dict[str, Any] | None = None,
) -> None:
    update_provenance(
        config.output_dir / "provenance.json",
        _base_provenance(config, cache, phase4),
        defaults={"fit_complete": False, "transfer_complete": False, "analysis_complete": False},
        updates=updates,
    )


def _read_activations(cache: RawActivationCache, indices: np.ndarray) -> np.ndarray:
    values = np.empty((len(indices), cache.width), dtype=np.float32)
    for start in range(0, len(indices), 256):
        batch = indices[start : start + 256]
        values[start : start + len(batch)] = cache.read(batch)
    if not bool(np.isfinite(values).all()):
        raise RuntimeError("Phase 1 raw activation cache contains non-finite values")
    return values


def _task_balanced_means(
    activations: np.ndarray, rows: pd.DataFrame, tasks: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    clean, attack = [], []
    for task in tasks:
        task_rows = rows.task.to_numpy(dtype=object) == task
        labels = rows.label.to_numpy(dtype=np.int64)
        for label, target in ((0, clean), (1, attack)):
            selected = activations[task_rows & (labels == label)]
            if selected.size == 0:
                raise RuntimeError(f"Missing raw training examples for task {task}, label {label}")
            target.append(selected.mean(axis=0, dtype=np.float64))
    return (
        np.mean(np.stack(clean), axis=0).astype(np.float32),
        np.mean(np.stack(attack), axis=0).astype(np.float32),
    )


def _append_macro_rates(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics]
    task_rates = metrics[(metrics.scope == "task") & metrics.metric.isin(["tpr", "fpr"])]
    extra = []
    for (detector, metric), values in task_rates.groupby(["detector", "metric"], sort=True):
        extra.append(
            {
                "detector": detector,
                "scope": "macro",
                "task": None,
                "task_display": "Macro",
                "metric": metric,
                "value": float(values.value.mean()),
                "threshold": float(values.threshold.iloc[0]),
                "n": int(values.n.sum()),
            }
        )
    if extra:
        rows.append(pd.DataFrame(extra))
    return pd.concat(rows, ignore_index=True)


def fit(config: RawBaselineConfig) -> Path:
    config.validate(require_roots=False)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_raw_activation_cache(config)
    phase4 = _reference_provenance(config)
    if phase4["phase1_run_id"] != cache.handoff.metadata["run_id"]:
        raise RuntimeError("Phase 4 does not descend from the supplied Phase 1 run")
    FrozenDetectors.load(config.jspace_phase3_dir, cache.handoff.metadata)
    _provenance(config, cache, phase4)
    current = read_json(config.output_dir / "provenance.json")
    if current.get("fit_complete") is True:
        detectors = RawDetectors.load(config.output_dir, cache.handoff.metadata)
        if detectors.mean["raw_baseline_config_sha256"] != config.identity_hash():
            raise RuntimeError("Frozen raw detectors do not match this configuration")
        required = (
            "raw_validation_scores.parquet",
            "raw_validation_metrics.csv",
            "raw_mean_detector.pt",
            "raw_logistic_detector.pt",
        )
        if any(not (config.output_dir / name).is_file() for name in required):
            raise RuntimeError("Raw fit provenance is complete but an artifact is missing")
        print(f"Raw detector fit already complete: {config.output_dir}")
        return config.output_dir

    examples = pd.DataFrame(cache.handoff.examples).set_index("example_index", drop=False)
    train_indices = examples.index[examples.split == "train"].to_numpy(dtype=np.int64)
    validation_indices = examples.index[examples.split == "validation"].to_numpy(dtype=np.int64)
    train = examples.loc[train_indices].copy()
    validation = examples.loc[validation_indices].copy()
    train_x = _read_activations(cache, train_indices)
    validation_x = _read_activations(cache, validation_indices)
    train_y = train.label.to_numpy(dtype=np.int64)
    validation_y = validation.label.to_numpy(dtype=np.int64)
    validation_tasks = validation.task.to_numpy(dtype=object)

    mu_clean, mu_attack = _task_balanced_means(train_x, train, config.phase1.tasks)
    direction = mu_attack - mu_clean
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Raw mean direction has zero or non-finite norm")
    unit = direction / norm
    mean_scores = np.asarray((validation_x - mu_clean) @ unit, dtype=np.float64)

    logistic = LogisticRegression(
        penalty=config.penalty,
        C=config.regularization_c,
        solver=config.solver,
        fit_intercept=config.fit_intercept,
        class_weight=config.class_weight,
        random_state=config.random_state,
        max_iter=config.max_iter,
        tol=config.tol,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        logistic.fit(train_x, train_y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Raw logistic regression did not converge")
    if logistic.classes_.tolist() != [0, 1]:
        raise RuntimeError("Raw logistic regression did not fit the expected labels")
    logistic_scores = np.asarray(logistic.decision_function(validation_x), dtype=np.float64)
    mean_threshold, _ = select_threshold(mean_scores, validation_y, validation_tasks)
    logistic_threshold, _ = select_threshold(logistic_scores, validation_y, validation_tasks)

    columns = ["example_index", "pair_id", "task", "task_display", "condition", "label"]
    scores = validation.reset_index(drop=True)[columns].copy()
    scores.insert(0, "example_id", scores.pair_id.astype(str) + ":" + scores.condition.astype(str))
    scores["mean_score"] = mean_scores
    scores["logistic_score"] = logistic_scores
    scores["mean_prediction"] = mean_scores >= mean_threshold
    scores["logistic_prediction"] = logistic_scores >= logistic_threshold
    atomic_write_parquet(config.output_dir / "raw_validation_scores.parquet", scores)
    thresholds = {"mean": mean_threshold, "logistic": logistic_threshold}
    metrics = _append_macro_rates(compute_metrics(scores, thresholds, config.phase1.tasks))
    atomic_write_csv(config.output_dir / "raw_validation_metrics.csv", metrics)

    common = {
        "schema_version": 1,
        "artifact_type": "raw_residual_detector",
        "frozen": True,
        "phase1_run_id": cache.handoff.metadata["run_id"],
        "phase1_config_sha256": cache.handoff.metadata["config_sha256"],
        "manifest_sha256": cache.handoff.metadata["manifest_sha256"],
        "raw_baseline_config_sha256": config.identity_hash(),
        "selected_layer": cache.handoff.selected_layer,
        "raw_width": cache.width,
        "training_examples": len(train_indices),
        "model": config.phase1.scientific_dict()["model"],
    }
    atomic_torch_save(
        config.output_dir / "raw_mean_detector.pt",
        {
            **common,
            "detector": "mean",
            "mu_clean": torch.from_numpy(mu_clean.copy()),
            "d_raw": torch.from_numpy(direction.copy()),
            "d_unit": torch.from_numpy(unit.copy()),
            "d_norm": norm,
            "threshold": mean_threshold,
        },
    )
    atomic_torch_save(
        config.output_dir / "raw_logistic_detector.pt",
        {
            **common,
            "detector": "logistic",
            "weights": torch.from_numpy(np.asarray(logistic.coef_[0], dtype=np.float64).copy()),
            "intercept": float(logistic.intercept_[0]),
            "threshold": logistic_threshold,
            "parameter_count": cache.width + 1,
            "settings": config.scientific_dict()["logistic"],
        },
    )
    load_raw_detector(config.output_dir / "raw_mean_detector.pt")
    load_raw_detector(config.output_dir / "raw_logistic_detector.pt")
    _provenance(
        config,
        cache,
        phase4,
        updates={
            "fit_complete": True,
            "fit_packages": package_versions(PACKAGES),
            "raw_detector_artifacts": {
                "mean_sha256": sha256_file(config.output_dir / "raw_mean_detector.pt"),
                "logistic_sha256": sha256_file(config.output_dir / "raw_logistic_detector.pt"),
            },
        },
    )
    print(f"Raw detector fit complete: {config.output_dir}")
    return config.output_dir


def _reference_records(config: RawBaselineConfig, name: str) -> dict[str, dict[str, Any]]:
    path = config.jspace_phase4_dir / f"{name}_records.jsonl"
    rows = read_resumable_jsonl(path)
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in values:
            raise RuntimeError(f"Invalid completed Phase 4 record at {path}")
        values[case_id] = row
    if not values:
        raise RuntimeError(f"Completed Phase 4 record stream is empty: {path}")
    return values


def _record_identity(
    config: RawBaselineConfig, cache: RawActivationCache, phase4: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "raw_baseline_config_sha256": config.identity_hash(),
        "phase1_run_id": cache.handoff.metadata["run_id"],
        "raw_mean_detector_sha256": sha256_file(config.output_dir / "raw_mean_detector.pt"),
        "raw_logistic_detector_sha256": sha256_file(config.output_dir / "raw_logistic_detector.pt"),
        "jspace_phase4_run_id": phase4["run_id"],
        "bipia_test_manifest_sha256": phase4["bipia_test_manifest_sha256"],
    }


def transfer(config: RawBaselineConfig) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("The raw baseline transfer stage requires a CUDA GPU")
    config.validate(require_roots=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_raw_activation_cache(config)
    phase4 = _reference_provenance(config)
    provenance = read_json(config.output_dir / "provenance.json")
    if provenance.get("fit_complete") is not True:
        raise RuntimeError("Raw detectors are not frozen; run --stage fit first")
    verify_checkout(config.agentdojo_root, config.agentdojo_revision, "AgentDojo")
    verify_checkout(config.injecagent_root, config.injecagent_revision, "InjecAgent")
    detectors = RawDetectors.load(config.output_dir, cache.handoff.metadata)
    if detectors.mean["raw_baseline_config_sha256"] != config.identity_hash():
        raise RuntimeError("Frozen raw detectors do not match this configuration")
    identity = _record_identity(config, cache, phase4)
    references = {name: _reference_records(config, name) for name in RECORD_NAMES}
    completed = {
        name: completed_records(config.output_dir / f"{name}_raw_records.jsonl", identity)
        for name in RECORD_NAMES
    }

    tokenizer = load_tokenizer(config.phase1)
    model = HuggingFaceModelAdapter.load(config.phase1, tokenizer)
    if model.hidden_width != cache.width:
        raise RuntimeError("Raw detector width does not match the pinned model")
    phase4_config = config.phase4_config(config.jspace_phase4_dir)
    bipia.generate(
        phase4_config,
        model,
        detectors,
        completed["bipia"],
        identity,
        output_path=config.output_dir / "bipia_raw_records.jsonl",
        reference_records=references["bipia"],
        capture_only=True,
    )
    agentdojo.generate(
        phase4_config,
        model,
        detectors,
        completed["agentdojo"],
        identity,
        output_path=config.output_dir / "agentdojo_raw_records.jsonl",
        reference_records=references["agentdojo"],
        capture_only=True,
    )
    injecagent.generate(
        phase4_config,
        model,
        detectors,
        completed["injecagent"],
        identity,
        output_path=config.output_dir / "injecagent_raw_records.jsonl",
        reference_records=references["injecagent"],
        capture_only=True,
    )
    counts = {
        name: len(completed_records(config.output_dir / f"{name}_raw_records.jsonl", identity))
        for name in RECORD_NAMES
    }
    reference_counts = {name: len(rows) for name, rows in references.items()}
    if counts != reference_counts:
        raise RuntimeError("Raw transfer record counts do not match completed Phase 4")
    _provenance(
        config,
        cache,
        phase4,
        updates={
            "transfer_complete": True,
            "transfer_record_counts": counts,
            "transfer_gpu": cuda_metadata(model_input_device=str(model.input_device)),
            "transfer_packages": package_versions(PACKAGES),
        },
    )
    del model, tokenizer, detectors, cache
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Raw transfer capture complete: {config.output_dir}")
    return config.output_dir


def _add_agentdojo_balanced_accuracy(metrics: pd.DataFrame) -> pd.DataFrame:
    extra = []
    dojo = metrics[metrics.benchmark == "agentdojo"]
    for (scope, subgroup, detector), values in dojo.groupby(
        ["scope", "subgroup", "detector"], dropna=False, sort=True
    ):
        by_metric = dict(zip(values.metric, values.value, strict=False))
        if detector in {"mean", "logistic"} and {"tpr", "fpr"}.issubset(by_metric):
            extra.append(
                {
                    "benchmark": "agentdojo",
                    "scope": scope,
                    "subgroup": subgroup,
                    "detector": detector,
                    "metric": "balanced_accuracy",
                    "value": (float(by_metric["tpr"]) + 1 - float(by_metric["fpr"])) / 2,
                    "n": int(values.n.max()),
                    "threshold": float(values.threshold.dropna().iloc[0]),
                }
            )
    return pd.concat([metrics, pd.DataFrame(extra)], ignore_index=True) if extra else metrics


def _validation_metric_values(frame: pd.DataFrame, detector: str) -> dict[str, float]:
    task_values = {name: [] for name in ("auprc", "auroc", "balanced_accuracy", "tpr", "fpr")}
    threshold = float(frame[f"{detector}_threshold"].iloc[0])
    for _, subset in frame.groupby("task", sort=True):
        labels = subset.label.to_numpy(dtype=np.int64)
        scores = subset[f"{detector}_score"].to_numpy(dtype=float)
        predictions = scores >= threshold
        tpr = float(predictions[labels == 1].mean())
        fpr = float(predictions[labels == 0].mean())
        task_values["auprc"].append(float(average_precision_score(labels, scores)))
        task_values["auroc"].append(float(roc_auc_score(labels, scores)))
        task_values["tpr"].append(tpr)
        task_values["fpr"].append(fpr)
        task_values["balanced_accuracy"].append((tpr + 1 - fpr) / 2)
    return {key: float(np.mean(values)) for key, values in task_values.items()}


def _bipia_metric_values(frame: pd.DataFrame, detector: str) -> dict[str, float]:
    task_values = {name: [] for name in ("auprc", "auroc", "balanced_accuracy", "tpr", "fpr")}
    threshold = float(frame[f"{detector}_threshold"].iloc[0])
    for _, subset in frame.groupby("task", sort=True):
        attacks = subset[subset.condition == "attack"]
        controls = subset[subset.condition == "control"].drop_duplicates("bootstrap_context")
        matched_controls = controls.set_index("bootstrap_context").loc[
            attacks.bootstrap_context.tolist()
        ]
        matched_scores = np.concatenate(
            [
                attacks[f"{detector}_score"].to_numpy(dtype=float),
                matched_controls[f"{detector}_score"].to_numpy(dtype=float),
            ]
        )
        matched_labels = np.concatenate(
            [np.ones(len(attacks), dtype=int), np.zeros(len(matched_controls), dtype=int)]
        )
        raw = pd.concat([attacks, controls], ignore_index=True)
        raw_labels = (raw.condition == "attack").to_numpy(dtype=int)
        attack_scores = attacks[f"{detector}_score"].to_numpy(dtype=float)
        control_scores = controls[f"{detector}_score"].to_numpy(dtype=float)
        tpr = float((attack_scores >= threshold).mean())
        fpr = float((control_scores >= threshold).mean())
        task_values["auprc"].append(float(average_precision_score(matched_labels, matched_scores)))
        task_values["auroc"].append(
            float(roc_auc_score(raw_labels, raw[f"{detector}_score"].to_numpy(dtype=float)))
        )
        task_values["tpr"].append(tpr)
        task_values["fpr"].append(fpr)
        task_values["balanced_accuracy"].append((tpr + 1 - fpr) / 2)
    return {key: float(np.mean(values)) for key, values in task_values.items()}


def _agentdojo_metric_values(frame: pd.DataFrame, detector: str) -> dict[str, float]:
    clean = frame[(frame.condition == "control") & frame[f"{detector}_score"].notna()]
    attacked = frame[
        (frame.condition == "attack")
        & frame.injection_exposed.fillna(False)
        & frame[f"{detector}_score"].notna()
    ]
    tpr = float(attacked[f"{detector}_prediction"].mean())
    fpr = float(clean[f"{detector}_prediction"].mean())
    return {"tpr": tpr, "fpr": fpr, "balanced_accuracy": (tpr + 1 - fpr) / 2}


def _injecagent_metric_values(frame: pd.DataFrame, detector: str) -> dict[str, float]:
    return {"tpr": float(frame[f"{detector}_prediction"].mean())}


def _resample_stratified(
    frame: pd.DataFrame, strata: list[str], unit: str, rng: np.random.Generator
) -> pd.DataFrame:
    sampled = []
    for _, subset in frame.groupby(strata, dropna=False, sort=True):
        units = subset[unit].drop_duplicates().tolist()
        draws = rng.choice(units, size=len(units), replace=True)
        for draw_index, selected in enumerate(draws):
            rows = subset[subset[unit] == selected].copy()
            rows["bootstrap_context"] = f"{draw_index}:{selected}"
            sampled.append(rows)
    return pd.concat(sampled, ignore_index=True)


def _representation_metric_values(
    sample: pd.DataFrame,
    prefix: str,
    detector: str,
    metric_fn: Callable[[pd.DataFrame, str], dict[str, float]],
) -> dict[str, float]:
    target_columns = [
        f"{detector}_score",
        f"{detector}_prediction",
        f"{detector}_threshold",
    ]
    mapping = {f"{prefix}_{name}": name for name in target_columns}
    representation = sample.drop(columns=target_columns, errors="ignore").rename(
        columns=mapping
    )
    return metric_fn(representation, detector)


def _bootstrap_deltas(
    frame: pd.DataFrame,
    *,
    strata: list[str],
    unit: str,
    metric_fn: Callable[[pd.DataFrame, str], dict[str, float]],
    replicates: int,
    seed: int,
) -> dict[tuple[str, str], tuple[float, float]]:
    rng = np.random.default_rng(seed)
    values: dict[tuple[str, str], list[float]] = {}
    for _ in range(replicates):
        sample = _resample_stratified(frame, strata, unit, rng)
        for detector in ("mean", "logistic"):
            j_values = _representation_metric_values(sample, "jspace", detector, metric_fn)
            raw_values = _representation_metric_values(sample, "raw", detector, metric_fn)
            for metric in j_values:
                delta = (
                    raw_values[metric] - j_values[metric]
                    if metric == "fpr"
                    else j_values[metric] - raw_values[metric]
                )
                values.setdefault((detector, metric), []).append(delta)
    return {
        key: (float(np.quantile(items, 0.025)), float(np.quantile(items, 0.975)))
        for key, items in values.items()
    }


def _comparison_rows(
    dataset: str,
    point: dict[str, dict[str, dict[str, float]]],
    intervals: dict[tuple[str, str], tuple[float, float]],
) -> list[dict[str, Any]]:
    rows = []
    for detector in ("mean", "logistic"):
        for metric, jspace_value in point["jspace"][detector].items():
            raw_value = point["raw"][detector][metric]
            delta = raw_value - jspace_value if metric == "fpr" else jspace_value - raw_value
            low, high = intervals[(detector, metric)]
            conclusion = "better" if low > 0 else "worse" if high < 0 else "inconclusive"
            rows.append(
                {
                    "dataset": dataset,
                    "scope": "macro" if dataset in {"validation", "bipia"} else "overall",
                    "detector": detector,
                    "metric": metric,
                    "jspace_value": jspace_value,
                    "raw_value": raw_value,
                    "delta_favoring_jspace": delta,
                    "ci95_low": low,
                    "ci95_high": high,
                    "conclusion": conclusion,
                }
            )
    return rows


def _paired_frame(
    metadata: pd.DataFrame, jspace: pd.DataFrame, raw: pd.DataFrame, key: str
) -> pd.DataFrame:
    score_columns = [key]
    for detector in ("mean", "logistic"):
        score_columns += [f"{detector}_score", f"{detector}_prediction"]
    merged = metadata.merge(
        jspace[score_columns].rename(
            columns={name: f"jspace_{name}" for name in score_columns if name != key}
        ),
        on=key,
        validate="one_to_one",
    ).merge(
        raw[score_columns].rename(
            columns={name: f"raw_{name}" for name in score_columns if name != key}
        ),
        on=key,
        validate="one_to_one",
    )
    if len(merged) != len(metadata):
        raise RuntimeError("J-space and raw records do not have identical identities")
    return merged


def _plot_validation(config: RawBaselineConfig, comparison: pd.DataFrame) -> None:
    rows = comparison[
        (comparison.dataset == "validation")
        & comparison.metric.isin(["auprc", "auroc", "balanced_accuracy"])
    ]
    labels = [
        f"{detector}\n{metric}"
        for detector, metric in zip(rows.detector, rows.metric, strict=False)
    ]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - 0.18, rows.jspace_value, 0.36, label="J-space")
    axis.bar(x + 0.18, rows.raw_value, 0.36, label="Raw residual")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Task-macro validation metric")
    axis.set_title("Same-Layer Validation: J-Space vs Raw Residual")
    axis.legend()
    figure.tight_layout()
    atomic_save_figure(config.output_dir / "jspace_vs_raw_validation.png", figure, dpi=180)
    plt.close(figure)


def _plot_transfer(config: RawBaselineConfig, comparison: pd.DataFrame) -> None:
    wanted = {("bipia", "auprc"), ("agentdojo", "balanced_accuracy"), ("injecagent", "tpr")}
    rows = comparison[comparison.apply(lambda row: (row.dataset, row.metric) in wanted, axis=1)]
    labels = [
        f"{dataset}\n{detector}"
        for dataset, detector in zip(rows.dataset, rows.detector, strict=False)
    ]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - 0.18, rows.jspace_value, 0.36, label="J-space")
    axis.bar(x + 0.18, rows.raw_value, 0.36, label="Raw residual")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Prespecified benchmark metric")
    axis.set_title("Frozen Detector Transfer: J-Space vs Raw Residual")
    axis.legend()
    figure.tight_layout()
    atomic_save_figure(config.output_dir / "jspace_vs_raw_transfer.png", figure, dpi=180)
    plt.close(figure)


def analyze(config: RawBaselineConfig) -> Path:
    config.validate(require_roots=False)
    cache = load_raw_activation_cache(config)
    phase4 = _reference_provenance(config)
    provenance = read_json(config.output_dir / "provenance.json")
    if (
        provenance.get("fit_complete") is not True
        or provenance.get("transfer_complete") is not True
    ):
        raise RuntimeError("Raw fit and transfer must complete before analysis")
    detectors = RawDetectors.load(config.output_dir, cache.handoff.metadata)
    identity = _record_identity(config, cache, phase4)
    raw_records = {
        name: list(
            completed_records(config.output_dir / f"{name}_raw_records.jsonl", identity).values()
        )
        for name in RECORD_NAMES
    }
    reference_predictions = pd.read_parquet(config.jspace_phase4_dir / "phase4_predictions.parquet")
    raw_scores = pd.DataFrame([row for values in raw_records.values() for row in values])
    if set(raw_scores.case_id) != set(reference_predictions.case_id):
        raise RuntimeError("Raw transfer cases do not exactly match completed Phase 4")
    score_columns = [
        "case_id",
        "mean_score",
        "mean_prediction",
        "logistic_score",
        "logistic_prediction",
    ]
    predictions = reference_predictions.drop(columns=score_columns[1:]).merge(
        raw_scores[score_columns], on="case_id", validate="one_to_one"
    )
    for column in ("mean_prediction", "logistic_prediction"):
        predictions[column] = pd.array(predictions[column], dtype="boolean")
    raw_metrics = _add_agentdojo_balanced_accuracy(transfer_metrics(predictions, detectors))
    atomic_write_csv(config.output_dir / "raw_transfer_metrics.csv", raw_metrics)
    compact_columns = [
        "case_id",
        "context_id",
        "source_clean_case_id",
        "benchmark",
        "task",
        "subgroup",
        "condition",
        "attack_category",
        "attack_variant_id",
        "position",
        "injection_exposed",
        "mean_score",
        "mean_prediction",
        "logistic_score",
        "logistic_prediction",
    ]
    atomic_write_parquet(
        config.output_dir / "raw_transfer_predictions.parquet",
        predictions[[name for name in compact_columns if name in predictions]],
    )

    raw_validation = pd.read_parquet(config.output_dir / "raw_validation_scores.parquet")
    j_validation = pd.read_parquet(config.jspace_phase3_dir / "phase3_validation_scores.parquet")
    validation_metadata = j_validation[["example_id", "pair_id", "task", "condition", "label"]]
    validation = _paired_frame(validation_metadata, j_validation, raw_validation, "example_id")
    j_detectors = FrozenDetectors.load(config.jspace_phase3_dir, cache.handoff.metadata)
    for prefix, source in (("jspace", j_detectors), ("raw", detectors)):
        validation[f"{prefix}_mean_threshold"] = float(source.mean["threshold"])
        validation[f"{prefix}_logistic_threshold"] = float(source.logistic["threshold"])

    transfer_metadata = reference_predictions.drop(
        columns=["mean_score", "mean_prediction", "logistic_score", "logistic_prediction"]
    )
    transfer_frame = _paired_frame(transfer_metadata, reference_predictions, predictions, "case_id")
    for prefix, source in (("jspace", j_detectors), ("raw", detectors)):
        transfer_frame[f"{prefix}_mean_threshold"] = float(source.mean["threshold"])
        transfer_frame[f"{prefix}_logistic_threshold"] = float(source.logistic["threshold"])

    comparisons: list[dict[str, Any]] = []
    validation_intervals = _bootstrap_deltas(
        validation,
        strata=["task"],
        unit="pair_id",
        metric_fn=_validation_metric_values,
        replicates=config.bootstrap_replicates,
        seed=42,
    )
    validation_point = {representation: {} for representation in ("jspace", "raw")}
    for representation in validation_point:
        renamed = validation.rename(
            columns={
                f"{representation}_{detector}_{suffix}": f"{detector}_{suffix}"
                for detector in ("mean", "logistic")
                for suffix in ("score", "prediction", "threshold")
            }
        )
        validation_point[representation] = {
            detector: _validation_metric_values(renamed, detector)
            for detector in ("mean", "logistic")
        }
    comparisons += _comparison_rows("validation", validation_point, validation_intervals)

    benchmark_specs = {
        "bipia": (["task"], "context_id", _bipia_metric_values),
        "agentdojo": (["subgroup", "condition"], "case_id", _agentdojo_metric_values),
        "injecagent": (["subgroup"], "case_id", _injecagent_metric_values),
    }
    for benchmark, (strata, unit, metric_fn) in benchmark_specs.items():
        frame = transfer_frame[transfer_frame.benchmark == benchmark].copy()
        if benchmark == "bipia":
            frame["bootstrap_context"] = frame.context_id
        intervals = _bootstrap_deltas(
            frame,
            strata=strata,
            unit=unit,
            metric_fn=metric_fn,
            replicates=config.bootstrap_replicates,
            seed=42,
        )
        point: dict[str, dict[str, dict[str, float]]] = {"jspace": {}, "raw": {}}
        for representation in point:
            renamed = frame.rename(
                columns={
                    f"{representation}_{detector}_{suffix}": f"{detector}_{suffix}"
                    for detector in ("mean", "logistic")
                    for suffix in ("score", "prediction", "threshold")
                }
            )
            if benchmark == "bipia":
                renamed["bootstrap_context"] = renamed.context_id
            point[representation] = {
                detector: metric_fn(renamed, detector) for detector in ("mean", "logistic")
            }
        comparisons += _comparison_rows(benchmark, point, intervals)

    comparison = pd.DataFrame(comparisons)
    atomic_write_csv(config.output_dir / "jspace_vs_raw_comparison.csv", comparison)
    _plot_validation(config, comparison)
    _plot_transfer(config, comparison)
    _provenance(
        config,
        cache,
        phase4,
        updates={
            "analysis_complete": True,
            "analysis_packages": package_versions(PACKAGES),
            "artifacts": {
                name: name
                for name in (
                    "raw_validation_scores.parquet",
                    "raw_validation_metrics.csv",
                    "raw_transfer_predictions.parquet",
                    "raw_transfer_metrics.csv",
                    "jspace_vs_raw_comparison.csv",
                    "jspace_vs_raw_validation.png",
                    "jspace_vs_raw_transfer.png",
                )
            },
        },
    )
    print(f"Raw baseline analysis complete: {config.output_dir}")
    return config.output_dir


def run(config: RawBaselineConfig, stage: str) -> Path:
    if stage == "fit":
        return fit(config)
    if stage == "transfer":
        return transfer(config)
    if stage == "analyze":
        return analyze(config)
    if stage == "all":
        fit(config)
        transfer(config)
        return analyze(config)
    raise ValueError(f"Unknown raw baseline stage: {stage}")
