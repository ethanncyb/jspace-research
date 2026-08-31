from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from ..phase1.artifacts import Phase1Handoff, load_phase1_handoff
from ..phase1.data import TASK_DISPLAY
from ..phase1.jspace import batched, direction_scores, read_bfloat16_bits
from ..runtime import (
    atomic_save_figure,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_parquet,
    update_provenance,
)
from .artifacts import load_detector
from .config import Phase3Config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _run_id(config: Phase3Config, handoff: Phase1Handoff) -> str:
    return f"phase3-{config.identity_hash()[:12]}-{handoff.metadata['manifest_sha256'][:12]}"


def _base_provenance(config: Phase3Config, handoff: Phase1Handoff) -> dict[str, Any]:
    phase1 = config.phase1.scientific_dict()
    return {
        "schema_version": 1,
        "phase": 3,
        "run_id": _run_id(config, handoff),
        "phase1_run_id": handoff.metadata["run_id"],
        "phase1_config_sha256": handoff.metadata["config_sha256"],
        "phase3_config_sha256": config.identity_hash(),
        "manifest_sha256": handoff.metadata["manifest_sha256"],
        "selected_layer": handoff.selected_layer,
        "decomposition": handoff.metadata["decomposition"],
        "model": phase1["model"],
        "lens": phase1["lens"],
        "dependencies": phase1["dependencies"],
        "seed": config.phase1.seed,
        "device": "cpu",
        "dtypes": {
            "reconstruction": "bfloat16",
            "sparse_coefficients": "float32",
            "logistic_parameters": "float64",
        },
        "resolved_config": config.scientific_dict(),
    }


def build_sparse_features(
    handoff: Phase1Handoff,
    train_indices: np.ndarray,
    requested_indices: np.ndarray,
    feature_token_ids: np.ndarray | None = None,
) -> tuple[csr_matrix, np.ndarray]:
    """Build raw-coefficient CSR features using training-supported token IDs."""

    training_supports = np.asarray(handoff.support_ids[train_indices], dtype=np.int64)
    training_coefficients = np.asarray(handoff.coefficients[train_indices], dtype=np.float32)
    if not bool(np.isfinite(training_coefficients).all()) or bool(
        (training_coefficients < 0).any()
    ):
        raise RuntimeError("Phase 1 sparse coefficients must be finite and nonnegative")
    if bool(((training_supports < 0) & (training_coefficients != 0)).any()):
        raise RuntimeError("Padded Phase 1 supports must have zero coefficients")

    if feature_token_ids is None:
        feature_token_ids = np.unique(training_supports[training_supports >= 0])
    else:
        feature_token_ids = np.asarray(feature_token_ids, dtype=np.int64)
    if feature_token_ids.ndim != 1 or feature_token_ids.size == 0:
        raise RuntimeError("Training examples contain no sparse J-space features")
    if bool((feature_token_ids[1:] <= feature_token_ids[:-1]).any()):
        raise ValueError("Feature token IDs must be sorted and unique")

    supports = np.asarray(handoff.support_ids[requested_indices], dtype=np.int64)
    coefficients = np.asarray(handoff.coefficients[requested_indices], dtype=np.float32)
    if not bool(np.isfinite(coefficients).all()) or bool((coefficients < 0).any()):
        raise RuntimeError("Phase 1 sparse coefficients must be finite and nonnegative")
    if bool(((supports < 0) & (coefficients != 0)).any()):
        raise RuntimeError("Padded Phase 1 supports must have zero coefficients")

    row_ids = np.repeat(np.arange(len(requested_indices)), supports.shape[1])
    flat_supports = supports.reshape(-1)
    flat_coefficients = coefficients.reshape(-1)
    active = (flat_supports >= 0) & (flat_coefficients != 0)
    active_supports = flat_supports[active]
    columns = np.searchsorted(feature_token_ids, active_supports)
    within_range = columns < len(feature_token_ids)
    known = np.zeros(len(columns), dtype=bool)
    known[within_range] = (
        feature_token_ids[columns[within_range]] == active_supports[within_range]
    )
    matrix = csr_matrix(
        (
            flat_coefficients[active][known],
            (row_ids[active][known], columns[known]),
        ),
        shape=(len(requested_indices), len(feature_token_ids)),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix, feature_token_ids


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    tasks: np.ndarray,
) -> tuple[float, float]:
    """Maximize task-macro balanced accuracy; exact ties use the higher threshold."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    tasks = np.asarray(tasks, dtype=object)
    if scores.ndim != 1 or len(scores) != len(labels) or len(labels) != len(tasks):
        raise ValueError("Scores, labels, and tasks must be equal-length vectors")
    if not bool(np.isfinite(scores).all()) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires finite scores and both labels")
    task_names = sorted(set(tasks.tolist()))
    for task in task_names:
        if set(np.unique(labels[tasks == task])) != {0, 1}:
            raise ValueError(f"Task {task} must contain both labels")

    unique_scores = np.unique(scores)
    all_negative = float(np.nextafter(unique_scores[-1], np.inf))
    if not np.isfinite(all_negative):
        raise ValueError("Could not construct a finite all-negative threshold")
    candidates = np.append(unique_scores, all_negative)
    best_threshold = float(candidates[0])
    best_value = -np.inf
    for candidate in candidates:
        predictions = scores >= candidate
        task_values = []
        for task in task_names:
            task_mask = tasks == task
            task_labels = labels[task_mask]
            task_predictions = predictions[task_mask]
            tpr = float(task_predictions[task_labels == 1].mean())
            tnr = float((~task_predictions[task_labels == 0]).mean())
            task_values.append((tpr + tnr) / 2)
        value = float(np.mean(task_values))
        if value > best_value or (value == best_value and candidate > best_threshold):
            best_value = value
            best_threshold = float(candidate)
    return best_threshold, float(best_value)


def _mean_scores(handoff: Phase1Handoff, indices: np.ndarray) -> np.ndarray:
    scores = np.empty(len(indices), dtype=np.float32)
    clean_mean = handoff.direction["mu_clean"]
    unit_direction = handoff.direction["d_unit"]
    offset = 0
    for batch in batched(indices, 256):
        representations = read_bfloat16_bits(
            np.asarray(handoff.reconstruction[batch]).copy()
        )
        batch_scores = direction_scores(representations, clean_mean, unit_direction)
        scores[offset : offset + len(batch)] = batch_scores.numpy()
        offset += len(batch)
    return scores


def _rates(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float, float]:
    positives = labels == 1
    negatives = labels == 0
    tpr = float(predictions[positives].mean())
    fpr = float(predictions[negatives].mean())
    return (tpr + (1.0 - fpr)) / 2.0, tpr, fpr


def compute_metrics(
    scores: pd.DataFrame,
    thresholds: dict[str, float],
    tasks: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for detector in ("mean", "logistic"):
        threshold = thresholds[detector]
        macro_values: dict[str, list[float]] = {
            "auprc": [],
            "auroc": [],
            "balanced_accuracy": [],
        }
        for task in tasks:
            subset = scores[scores.task == task]
            labels = subset.label.to_numpy(dtype=np.int64)
            continuous = subset[f"{detector}_score"].to_numpy(dtype=np.float64)
            predictions = continuous >= threshold
            auprc = float(average_precision_score(labels, continuous))
            auroc = float(roc_auc_score(labels, continuous))
            balanced_accuracy, tpr, fpr = _rates(labels, predictions)
            values = {
                "auprc": auprc,
                "auroc": auroc,
                "balanced_accuracy": balanced_accuracy,
                "tpr": tpr,
                "fpr": fpr,
            }
            for metric, value in values.items():
                rows.append(
                    {
                        "detector": detector,
                        "scope": "task",
                        "task": task,
                        "task_display": TASK_DISPLAY[task],
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "n": len(subset),
                    }
                )
                if metric in macro_values:
                    macro_values[metric].append(value)
        for metric, values in macro_values.items():
            rows.append(
                {
                    "detector": detector,
                    "scope": "macro",
                    "task": None,
                    "task_display": "Macro",
                    "metric": metric,
                    "value": float(np.mean(values)),
                    "threshold": threshold,
                    "n": len(scores),
                }
            )
    return pd.DataFrame(rows)


def _save_comparison_plot(config: Phase3Config, metrics: pd.DataFrame) -> Path:
    auprc = metrics[metrics.metric == "auprc"]
    labels = [TASK_DISPLAY[task] for task in config.phase1.tasks] + ["Macro"]
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))
    for offset, detector in ((-width / 2, "mean"), (width / 2, "logistic")):
        detector_rows = auprc[auprc.detector == detector]
        values = []
        for task in config.phase1.tasks:
            task_row = detector_rows[
                (detector_rows.scope == "task") & (detector_rows.task == task)
            ]
            values.append(float(task_row.value.iloc[0]))
        values.append(float(detector_rows[detector_rows.scope == "macro"].value.iloc[0]))
        axis.bar(positions + offset, values, width, label=detector.capitalize())
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Validation AUPRC")
    axis.set_title("Phase 3 Detector Comparison")
    axis.legend()
    figure.tight_layout()
    path = config.output_dir / "phase3_detector_comparison.png"
    atomic_save_figure(path, figure, dpi=180)
    plt.close(figure)
    return path


def run(config: Phase3Config) -> Path:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    handoff = load_phase1_handoff(config.phase1_selected_path, config.phase1)
    provenance = _base_provenance(config, handoff)
    update_provenance(
        config.output_dir / "provenance.json",
        provenance,
        updates={"frozen": False},
    )

    examples = pd.DataFrame(handoff.examples).set_index("example_index", drop=False)
    train_indices = examples.index[examples.split == "train"].to_numpy(dtype=np.int64)
    validation_indices = examples.index[examples.split == "validation"].to_numpy(
        dtype=np.int64
    )
    train_labels = examples.loc[train_indices].label.to_numpy(dtype=np.int64)
    validation = examples.loc[validation_indices].copy()
    validation_labels = validation.label.to_numpy(dtype=np.int64)
    validation_tasks = validation.task.to_numpy(dtype=object)

    train_features, feature_token_ids = build_sparse_features(
        handoff, train_indices, train_indices
    )
    validation_features, _ = build_sparse_features(
        handoff,
        train_indices,
        validation_indices,
        feature_token_ids,
    )
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
        logistic.fit(train_features, train_labels)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Phase 3 logistic regression did not converge")
    if logistic.classes_.tolist() != [0, 1]:
        raise RuntimeError("Phase 3 logistic regression did not fit the expected labels")

    mean_scores = _mean_scores(handoff, validation_indices)
    logistic_scores = np.asarray(
        logistic.decision_function(validation_features), dtype=np.float64
    )
    mean_threshold, _ = select_threshold(
        mean_scores, validation_labels, validation_tasks
    )
    logistic_threshold, _ = select_threshold(
        logistic_scores, validation_labels, validation_tasks
    )
    thresholds = {"mean": mean_threshold, "logistic": logistic_threshold}

    score_columns = [
        "example_index",
        "pair_id",
        "task",
        "task_display",
        "condition",
        "label",
    ]
    scores = validation.reset_index(drop=True)[score_columns].copy()
    scores.insert(
        0,
        "example_id",
        scores.pair_id.astype(str) + ":" + scores.condition.astype(str),
    )
    scores["mean_score"] = mean_scores
    scores["logistic_score"] = logistic_scores
    scores["mean_prediction"] = mean_scores >= mean_threshold
    scores["logistic_prediction"] = logistic_scores >= logistic_threshold
    scores_path = config.output_dir / "phase3_validation_scores.parquet"
    atomic_write_parquet(scores_path, scores)

    metrics = compute_metrics(scores, thresholds, config.phase1.tasks)
    metrics_path = config.output_dir / "phase3_metrics.csv"
    atomic_write_csv(metrics_path, metrics)
    _save_comparison_plot(config, metrics)

    common_artifact = {
        "schema_version": 1,
        "phase": 3,
        "frozen": True,
        "phase1_run_id": handoff.metadata["run_id"],
        "phase3_config_sha256": config.identity_hash(),
        "selected_layer": handoff.selected_layer,
        "decomposition": handoff.metadata["decomposition"],
    }
    mean_path = config.output_dir / "mean_detector.pt"
    atomic_torch_save(
        mean_path,
        {
            **common_artifact,
            "detector": "mean",
            "mu_clean": handoff.direction["mu_clean"].float().cpu(),
            "d_raw": handoff.direction["d_raw"].float().cpu(),
            "d_unit": handoff.direction["d_unit"].float().cpu(),
            "d_norm": float(handoff.direction["d_norm"]),
            "threshold": mean_threshold,
        },
    )
    logistic_path = config.output_dir / "logistic_detector.pt"
    atomic_torch_save(
        logistic_path,
        {
            **common_artifact,
            "detector": "logistic",
            "feature_token_ids": torch.from_numpy(feature_token_ids.copy()),
            "weights": torch.from_numpy(np.asarray(logistic.coef_[0]).copy()),
            "intercept": float(logistic.intercept_[0]),
            "threshold": logistic_threshold,
            "settings": config.scientific_dict()["logistic"],
        },
    )
    load_detector(mean_path)
    load_detector(logistic_path)
    update_provenance(
        config.output_dir / "provenance.json",
        provenance,
        updates={"frozen": True},
    )
    print(
        json.dumps(
            {
                "mean_threshold": mean_threshold,
                "logistic_threshold": logistic_threshold,
            },
            indent=2,
        )
    )
    print(f"Phase 3 metrics: {metrics_path}")
    return metrics_path
