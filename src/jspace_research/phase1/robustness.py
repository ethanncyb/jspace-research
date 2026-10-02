from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..runtime import (
    atomic_save_figure,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_parquet,
    package_versions,
    read_json,
    sha256_file,
    update_provenance,
)
from .adapters import JacobianLensAdapter, validate_lens_for_layers
from .artifacts import load_selected_layer
from .cache import load_done
from .config import Phase1Config
from .data import TASK_DISPLAY, expand_examples, read_jsonl, validate_pair_manifest
from .jspace import (
    batched,
    build_normalized_dictionary,
    direction_scores,
    learn_task_balanced_direction,
    nonnegative_gradient_pursuit,
    read_bfloat16_bits,
    screened_nonnegative_pursuit,
    select_task_macro_balanced_threshold,
    threshold_metrics,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPARSITY_K = 25
TRAIN_PAIRS_PER_TASK = 200
VALIDATION_PAIRS_PER_TASK = 100
METHODS = ("screened_greedy", "gradient_pursuit")


def _relative_artifact(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Artifact path escapes the Phase 1 run directory: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_memmap(path: Path, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.memmap:
    expected = int(np.prod(shape)) * dtype.itemsize
    if not path.is_file() or path.stat().st_size != expected:
        raise RuntimeError(f"Cache file size does not match metadata: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _load_source(
    config: Phase1Config, selected_path: Path
) -> tuple[dict[str, Any], Path, pd.DataFrame, dict[str, Any], np.memmap]:
    selected, _ = load_selected_layer(selected_path)
    if selected.get("config_sha256") != config.identity_hash():
        raise RuntimeError("Phase 1 selection does not match the supplied configuration")
    root = selected_path.parent
    manifest_path = _relative_artifact(root, selected["artifacts"]["pair_manifest"])
    rows = read_jsonl(manifest_path)
    validate_pair_manifest(rows, config)
    examples = pd.DataFrame(expand_examples(rows)).set_index("example_index", drop=False)

    activation_info = selected["artifacts"]["activations"]
    activation_metadata = read_json(_relative_artifact(root, activation_info["metadata"]))
    expected_lens_sha256 = selected["resolved_config"]["lens"]["sha256"]
    if (
        activation_metadata.get("config_sha256") != selected["config_sha256"]
        or activation_metadata.get("manifest_sha256") != selected["manifest_sha256"]
        or activation_metadata.get("lens_sha256") != expected_lens_sha256
    ):
        raise RuntimeError("Phase 1 activation identity does not match the frozen handoff")
    count = int(activation_metadata["number_examples"])
    layers = [int(value) for value in activation_metadata["layers"]]
    width = int(activation_metadata["d_model"])
    if count != len(examples):
        raise RuntimeError("Phase 1 manifest and activation counts do not match")
    done = load_done(_relative_artifact(root, activation_info["completion"]), count)
    if not bool(done.all()):
        raise RuntimeError("Phase 1 activation cache is incomplete")
    activations = _read_memmap(
        _relative_artifact(root, activation_info["residuals"]),
        np.dtype(np.uint16),
        (count, len(layers), width),
    )
    return selected, root, examples, activation_metadata, activations


def _validation_rows(
    frame: pd.DataFrame,
    *,
    layer: int,
    threshold: float,
    tasks: tuple[str, ...],
    method: str,
) -> list[dict[str, Any]]:
    validation = frame[frame.split == "validation"]
    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        subset = validation[validation.task == task]
        task_rows.append(
            {
                "layer": layer,
                "method": method,
                "sparsity_k": SPARSITY_K,
                "scope": "task",
                "task": task,
                "task_display": TASK_DISPLAY[task],
                "threshold": threshold,
                "n": len(subset),
                **threshold_metrics(subset.label, subset.score, threshold),
            }
        )
    metric_names = ("auprc", "auroc", "balanced_accuracy", "tpr", "fpr")
    macro = {name: float(np.mean([row[name] for row in task_rows])) for name in metric_names}
    return task_rows + [
        {
            "layer": layer,
            "method": method,
            "sparsity_k": SPARSITY_K,
            "scope": "macro",
            "task": None,
            "task_display": "Macro",
            "threshold": threshold,
            "n": int(sum(row["n"] for row in task_rows)),
            **macro,
        }
    ]


def _sample_indices(examples: pd.DataFrame, tasks: tuple[str, ...]) -> np.ndarray:
    rng = np.random.default_rng(42)
    quotas = {
        "train": TRAIN_PAIRS_PER_TASK,
        "validation": VALIDATION_PAIRS_PER_TASK,
    }
    selected_indices: list[int] = []
    for split, quota in quotas.items():
        for task in tasks:
            subset = examples[(examples.split == split) & (examples.task == task)]
            pair_ids = np.asarray(sorted(subset.pair_id.unique().tolist()), dtype=object)
            count = min(quota, len(pair_ids))
            chosen = set(rng.choice(pair_ids, size=count, replace=False).tolist())
            selected_indices.extend(subset[subset.pair_id.isin(chosen)].index.tolist())
    return np.asarray(sorted(selected_indices), dtype=np.int64)


def _comparison_layers(selected_layer: int, run_layers: list[int]) -> list[int]:
    layers = [
        layer for layer in range(selected_layer - 2, selected_layer + 3) if layer in run_layers
    ]
    if not layers:
        raise RuntimeError("The Phase 1 cache does not include the selected layer")
    return layers


def _cache_paths(output_dir: Path, method: str, layer: int, sparsity_k: int) -> tuple[Path, Path]:
    stem = output_dir / "cache" / method / f"layer_{layer:03d}_k{sparsity_k:03d}"
    return stem.with_suffix(".json"), stem.with_suffix(".pt")


def _load_or_compute_reconstruction(
    *,
    output_dir: Path,
    method: str,
    layer: int,
    sparsity_k: int,
    subset_indices: np.ndarray,
    subset_hash: str,
    layer_position: int,
    activations: np.memmap,
    dictionary: torch.Tensor,
    config: Phase1Config,
    selected: dict[str, Any],
) -> dict[str, torch.Tensor]:
    metadata_path, cache_path = _cache_paths(output_dir, method, layer, sparsity_k)
    identity = {
        "cache_schema_version": 2,
        "source_run_id": selected["run_id"],
        "source_manifest_sha256": selected["manifest_sha256"],
        "subset_sha256": subset_hash,
        "method": method,
        "layer": layer,
        "layer_position": layer_position,
        "sparsity_k": sparsity_k,
        "screen_candidates": (config.screen_candidates if method == "screened_greedy" else None),
        "example_count": len(subset_indices),
        "width": int(activations.shape[2]),
    }
    if metadata_path.exists():
        if read_json(metadata_path) != identity:
            raise RuntimeError(f"Robustness cache identity mismatch: {metadata_path}")
        if not cache_path.is_file():
            raise RuntimeError(f"Robustness cache is incomplete: {cache_path}")
        value = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(value, dict)
            or value.get("example_indices").tolist() != subset_indices.tolist()
        ):
            raise RuntimeError(f"Robustness cache contents are invalid: {cache_path}")
        return value

    reconstructions: list[torch.Tensor] = []
    supports: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []
    batches = list(batched(subset_indices, config.decomposition_batch_size))
    for indices in tqdm(batches, desc=f"{method} L{layer} k={sparsity_k}"):
        hidden = read_bfloat16_bits(np.asarray(activations[indices, layer_position, :]).copy())
        if method == "screened_greedy":
            reconstructed, token_ids, weights = screened_nonnegative_pursuit(
                hidden,
                dictionary,
                sparsity_k=sparsity_k,
                screen_candidates=config.screen_candidates,
            )
        else:
            reconstructed, token_ids, weights = nonnegative_gradient_pursuit(
                hidden, dictionary, sparsity_k=sparsity_k
            )
        reconstructions.append(reconstructed.to(torch.bfloat16))
        supports.append(token_ids.to(torch.int32))
        coefficients.append(weights.float())
    value = {
        "example_indices": torch.from_numpy(subset_indices.copy()),
        "reconstruction": torch.cat(reconstructions),
        "support_ids": torch.cat(supports),
        "coefficients": torch.cat(coefficients),
    }
    atomic_torch_save(cache_path, value)
    from ..runtime import atomic_write_json

    atomic_write_json(metadata_path, identity)
    return value


def _reconstruction_quality(
    hidden: torch.Tensor, reconstruction: torch.Tensor
) -> dict[str, np.ndarray]:
    hidden = hidden.float()
    reconstruction = reconstruction.float()
    hidden_norm = hidden.norm(dim=1).clamp_min(1e-12)
    error_norm = (hidden - reconstruction).norm(dim=1)
    return {
        "normalized_error": (error_norm / hidden_norm).numpy(),
        "cosine_similarity": F.cosine_similarity(hidden, reconstruction, dim=1).numpy(),
        "variance_explained": (1.0 - error_norm.square() / hidden_norm.square()).numpy(),
    }


def _support_jaccard(
    left_ids: torch.Tensor,
    left_coefficients: torch.Tensor,
    right_ids: torch.Tensor,
    right_coefficients: torch.Tensor,
) -> np.ndarray:
    values: list[float] = []
    rows = zip(
        left_ids.tolist(),
        left_coefficients.tolist(),
        right_ids.tolist(),
        right_coefficients.tolist(),
        strict=True,
    )
    for left_row, left_weights, right_row, right_weights in rows:
        left_set = {
            int(token_id)
            for token_id, weight in zip(left_row, left_weights, strict=True)
            if token_id >= 0 and weight > 0
        }
        right_set = {
            int(token_id)
            for token_id, weight in zip(right_row, right_weights, strict=True)
            if token_id >= 0 and weight > 0
        }
        union = left_set | right_set
        values.append(1.0 if not union else len(left_set & right_set) / len(union))
    return np.asarray(values, dtype=np.float64)


def _plot_density_grid(output_dir: Path, scores: pd.DataFrame, layers: list[int]) -> None:
    validation = scores[scores.split == "validation"]
    low = float(validation.score.min())
    high = float(validation.score.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    bins = np.linspace(low, high, 41)
    method_titles = {
        "screened_greedy": "Screened greedy",
        "gradient_pursuit": "Gradient pursuit",
    }
    figure, axes = plt.subplots(
        len(layers), len(METHODS), figsize=(12, 3.1 * len(layers)), squeeze=False
    )
    for row_index, layer in enumerate(layers):
        for column_index, method in enumerate(METHODS):
            axis = axes[row_index, column_index]
            subset = validation[(validation.layer == layer) & (validation.method == method)]
            for condition in ("control", "attack"):
                axis.hist(
                    subset[subset.condition == condition].score,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    label=condition,
                )
            axis.set_xlim(low, high)
            axis.set_title(f"{method_titles[method]}, L{layer}")
            if row_index == len(layers) - 1:
                axis.set_xlabel("Clean-to-attack direction score")
            if column_index == 0:
                axis.set_ylabel("Density")
            if row_index == 0:
                axis.legend()
    figure.suptitle("Phase 1 Reconstruction Robustness: All Tasks", y=1.0)
    figure.tight_layout()
    atomic_save_figure(output_dir / "reconstruction_density_by_layer.png", figure, dpi=180)
    plt.close(figure)


def _base_provenance(
    config: Phase1Config,
    selected: dict[str, Any],
    selected_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "study": "phase1_posthoc_reconstruction_robustness",
        "source_phase1_run_id": selected["run_id"],
        "source_selected_layer_sha256": sha256_file(selected_path),
        "source_manifest_sha256": selected["manifest_sha256"],
        "config_sha256": config.identity_hash(),
        "seed": 42,
        "test_split_used": False,
        "sparsity_k": SPARSITY_K,
        "train_pairs_per_task": TRAIN_PAIRS_PER_TASK,
        "validation_pairs_per_task": VALIDATION_PAIRS_PER_TASK,
        "methods": list(METHODS),
        "packages": package_versions(("jspace-research", "jlens", "torch")),
    }


def run_reconstruction_comparison(
    config: Phase1Config,
    selected_path: Path,
    output_dir: Path,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("The reconstruction robustness study requires a CUDA GPU")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, root, examples, activation_metadata, activations = _load_source(config, selected_path)
    run_layers = [int(value) for value in activation_metadata["layers"]]
    comparison_layers = _comparison_layers(int(selected["selected_layer"]), run_layers)
    subset_indices = _sample_indices(examples, config.tasks)
    subset_hash = hashlib.sha256(
        json.dumps(subset_indices.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    subset = examples.loc[subset_indices].copy()

    lens = JacobianLensAdapter.load(config)
    validate_lens_for_layers(lens, int(activation_metadata["d_model"]), comparison_layers)
    unembedding = torch.load(
        _relative_artifact(root, selected["artifacts"]["activations"]["unembedding"]),
        map_location="cpu",
        weights_only=True,
    )
    if unembedding.ndim != 2 or unembedding.shape[1] != int(activation_metadata["d_model"]):
        raise RuntimeError("Phase 1 unembedding width does not match the activation cache")

    device = torch.device("cuda:0")
    cached: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    for layer in comparison_layers:
        layer_position = run_layers.index(layer)
        dictionary = build_normalized_dictionary(
            jacobian=lens.jacobian(layer),
            unembedding=unembedding,
            layer=layer,
            device=device,
            chunk_size=config.dictionary_chunk_size,
        )
        for method in METHODS:
            cached[(method, layer)] = _load_or_compute_reconstruction(
                output_dir=output_dir,
                method=method,
                layer=layer,
                sparsity_k=SPARSITY_K,
                subset_indices=subset_indices,
                subset_hash=subset_hash,
                layer_position=layer_position,
                activations=activations,
                dictionary=dictionary,
                config=config,
                selected=selected,
            )
        del dictionary
        gc.collect()
        torch.cuda.empty_cache()

    metric_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    split_values = subset.split.to_numpy()
    task_values = subset.task.to_numpy()
    for layer in comparison_layers:
        layer_position = run_layers.index(layer)
        hidden = read_bfloat16_bits(
            np.asarray(activations[subset_indices, layer_position, :]).copy()
        )
        jaccard = _support_jaccard(
            cached[("screened_greedy", layer)]["support_ids"],
            cached[("screened_greedy", layer)]["coefficients"],
            cached[("gradient_pursuit", layer)]["support_ids"],
            cached[("gradient_pursuit", layer)]["coefficients"],
        )
        for method in METHODS:
            reconstruction = cached[(method, layer)]["reconstruction"].float()
            quality = _reconstruction_quality(hidden, reconstruction)
            train_mask = split_values == "train"
            direction = learn_task_balanced_direction(
                reconstruction[train_mask],
                subset.loc[train_mask, "label"].to_numpy(),
                subset.loc[train_mask, "task"].tolist(),
            )
            scores = direction_scores(
                reconstruction, direction["mu_clean"], direction["d_unit"]
            ).numpy()
            frame = subset[
                ["example_index", "pair_id", "split", "task", "condition", "label"]
            ].copy()
            frame["layer"] = layer
            frame["method"] = method
            frame["sparsity_k"] = SPARSITY_K
            frame["score"] = scores
            training = frame[frame.split == "train"]
            threshold, training_ba = select_task_macro_balanced_threshold(
                training.label, training.score, training.task.tolist()
            )
            rows = _validation_rows(
                frame,
                layer=layer,
                threshold=threshold,
                tasks=config.tasks,
                method=method,
            )
            for row in rows:
                scope_mask = (
                    task_values == row["task"]
                    if row["scope"] == "task"
                    else np.ones(len(subset), dtype=bool)
                )
                for name, values in quality.items():
                    row[f"train_{name}"] = float(
                        values[scope_mask & (split_values == "train")].mean()
                    )
                    row[f"validation_{name}"] = float(
                        values[scope_mask & (split_values == "validation")].mean()
                    )
                row["training_macro_balanced_accuracy"] = training_ba
                row["greedy_gradient_support_jaccard"] = float(jaccard[scope_mask].mean())
            metric_rows.extend(rows)
            score_frames.append(frame)

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(score_frames, ignore_index=True)
    metrics_path = output_dir / "reconstruction_robustness.csv"
    scores_path = output_dir / "reconstruction_validation_scores.parquet"
    atomic_write_csv(metrics_path, metrics)
    atomic_write_parquet(scores_path, scores[scores.split == "validation"].reset_index(drop=True))
    _plot_density_grid(output_dir, scores, comparison_layers)
    split_pair_counts = {
        split: int(subset[subset.split == split].pair_id.nunique())
        for split in ("train", "validation")
    }
    update_provenance(
        output_dir / "provenance.json",
        _base_provenance(config, selected, selected_path),
        defaults={"comparison_complete": False},
        updates={
            "comparison_complete": True,
            "comparison_layers": comparison_layers,
            "sample_pair_counts": split_pair_counts,
            "sample_example_indices_sha256": subset_hash,
            "sample_example_count": len(subset_indices),
            "gradient_pursuit": {
                "dictionary_search": "full",
                "selection": "largest_positive_residual_correlation",
                "update": "active_support_gradient_with_exact_line_search",
                "nonnegativity": "step_clipped_to_feasible_coefficients",
            },
        },
    )
    print(f"Reconstruction robustness complete: {metrics_path}")
    return metrics_path


def run(
    config: Phase1Config,
    selected_path: str | Path,
    output_dir: str | Path,
) -> None:
    selected = Path(selected_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == selected.parent or selected.parent in output.parents:
        raise ValueError("Robustness output must not be inside the frozen Phase 1 directory")
    run_reconstruction_comparison(config, selected, output)
