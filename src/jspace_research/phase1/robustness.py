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

K_VALUES = (10, 25, 50)
PILOT_PAIRS_PER_TASK_SPLIT = 50
METHODS = ("screened_greedy", "gradient_pursuit")


def _relative_artifact(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Artifact path escapes the Phase 1 run directory: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _decomposition_paths(root: Path, layer: int) -> dict[str, Path]:
    stem = root / "cache" / "decompositions" / f"layer_{layer:03d}"
    return {
        "metadata": stem.with_suffix(".json"),
        "reconstruction": stem.with_name(stem.name + "_bfloat16.dat"),
        "done": stem.with_name(stem.name + "_done.npy"),
    }


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


def _load_layer_reconstruction(
    root: Path,
    selected: dict[str, Any],
    layer: int,
    number_examples: int,
    width: int,
) -> np.memmap:
    paths = _decomposition_paths(root, layer)
    metadata = read_json(paths["metadata"])
    expected_lens_sha256 = selected["resolved_config"]["lens"]["sha256"]
    if (
        metadata.get("config_sha256") != selected["config_sha256"]
        or metadata.get("manifest_sha256") != selected["manifest_sha256"]
        or metadata.get("lens_sha256") != expected_lens_sha256
        or int(metadata.get("sparsity_k", -1)) != 25
        or int(metadata.get("screen_candidates", -1)) != 512
        or int(metadata.get("layer", -1)) != layer
        or metadata.get("reconstruction_shape") != [number_examples, width]
    ):
        raise RuntimeError(f"Phase 1 decomposition identity mismatch at layer {layer}")
    done = load_done(paths["done"], number_examples)
    if not bool(done.all()):
        raise RuntimeError(f"Phase 1 decomposition is incomplete at layer {layer}")
    return _read_memmap(
        paths["reconstruction"], np.dtype(np.uint16), (number_examples, width)
    )


def _scores_for_layer(
    root: Path,
    layer: int,
    examples: pd.DataFrame,
    reconstruction: np.memmap,
) -> pd.DataFrame:
    artifact_path = root / "layer_artifacts" / f"layer_{layer:03d}.pt"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing Phase 1 direction artifact: {artifact_path}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    required = {"layer", "mu_clean", "d_raw", "d_norm", "d_unit"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise RuntimeError(f"Direction artifact is incomplete at layer {layer}")
    if (
        int(artifact.get("layer", -1)) != layer
        or artifact["mu_clean"].shape != artifact["d_unit"].shape
        or artifact["d_unit"].numel() != reconstruction.shape[1]
    ):
        raise RuntimeError(f"Direction artifact layer mismatch at layer {layer}")

    records: list[dict[str, Any]] = []
    all_indices = examples.index.to_numpy(dtype=np.int64)
    for indices in batched(all_indices, 128):
        representations = read_bfloat16_bits(np.asarray(reconstruction[indices]).copy())
        scores = direction_scores(
            representations, artifact["mu_clean"], artifact["d_unit"]
        ).numpy()
        for index, score in zip(indices.tolist(), scores.tolist(), strict=True):
            row = examples.loc[index]
            records.append(
                {
                    "layer": layer,
                    "example_index": index,
                    "split": row.split,
                    "task": row.task,
                    "condition": row.condition,
                    "label": int(row.label),
                    "score": float(score),
                }
            )
    return pd.DataFrame(records)


def _validation_rows(
    frame: pd.DataFrame,
    *,
    layer: int,
    threshold: float,
    tasks: tuple[str, ...],
    method: str | None = None,
    sparsity_k: int | None = None,
) -> list[dict[str, Any]]:
    validation = frame[frame.split == "validation"]
    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        subset = validation[validation.task == task]
        metrics = threshold_metrics(subset.label, subset.score, threshold)
        task_rows.append(
            {
                "layer": layer,
                "method": method,
                "sparsity_k": sparsity_k,
                "scope": "task",
                "task": task,
                "task_display": TASK_DISPLAY[task],
                "threshold": threshold,
                "n": len(subset),
                **metrics,
            }
        )
    macro = {
        key: float(np.mean([row[key] for row in task_rows]))
        for key in ("auprc", "auroc", "balanced_accuracy", "tpr", "fpr")
    }
    return task_rows + [
        {
            "layer": layer,
            "method": method,
            "sparsity_k": sparsity_k,
            "scope": "macro",
            "task": None,
            "task_display": "Macro",
            "threshold": threshold,
            "n": int(sum(row["n"] for row in task_rows)),
            **macro,
        }
    ]


def _select_ba_layer(metrics: pd.DataFrame) -> int:
    macro = metrics[metrics.scope == "macro"]
    return int(
        macro.sort_values(
            ["balanced_accuracy", "layer"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0].layer
    )


def _plot_metric_comparison(
    output_dir: Path,
    metrics: pd.DataFrame,
    auprc_layer: int,
    ba_layer: int,
) -> None:
    macro = metrics[metrics.scope == "macro"].sort_values("layer")
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(macro.layer, macro.auprc, marker="o", label="Validation macro AUPRC")
    axis.plot(
        macro.layer,
        macro.balanced_accuracy,
        marker="o",
        label="Validation macro balanced accuracy",
    )
    axis.axvline(auprc_layer, linestyle="--", label=f"AUPRC selection L{auprc_layer}")
    axis.axvline(ba_layer, linestyle=":", label=f"BA comparison L{ba_layer}")
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("J-lens layer")
    axis.set_ylabel("Validation metric")
    axis.set_title("Phase 1 Layer-Selection Robustness")
    axis.legend()
    figure.tight_layout()
    atomic_save_figure(output_dir / "layer_metric_comparison.png", figure, dpi=180)
    plt.close(figure)


def _plot_density_comparison(
    output_dir: Path,
    scores: pd.DataFrame,
    tasks: tuple[str, ...],
    auprc_layer: int,
    ba_layer: int,
) -> None:
    layers = list(dict.fromkeys((auprc_layer, ba_layer)))
    validation = scores[(scores.split == "validation") & scores.layer.isin(layers)]
    low = float(validation.score.min())
    high = float(validation.score.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    bins = np.linspace(low, high, 41)
    scopes: list[tuple[str, str | None]] = [("All tasks", None)] + [
        (TASK_DISPLAY[task], task) for task in tasks
    ]
    figure, axes = plt.subplots(
        len(scopes), len(layers), figsize=(5.5 * len(layers), 3.2 * len(scopes)), squeeze=False
    )
    for row_index, (label, task) in enumerate(scopes):
        for column_index, layer in enumerate(layers):
            axis = axes[row_index, column_index]
            subset = validation[validation.layer == layer]
            if task is not None:
                subset = subset[subset.task == task]
            for condition in ("control", "attack"):
                axis.hist(
                    subset[subset.condition == condition].score,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    label=condition,
                )
            axis.set_xlim(low, high)
            axis.set_title(f"{label}, L{layer}")
            if row_index == len(scopes) - 1:
                axis.set_xlabel("Clean-to-attack direction score")
            if column_index == 0:
                axis.set_ylabel("Density")
            if row_index == 0:
                axis.legend()
    figure.tight_layout()
    atomic_save_figure(
        output_dir / "selected_layer_density_comparison.png", figure, dpi=180
    )
    plt.close(figure)


def _base_provenance(
    config: Phase1Config,
    selected: dict[str, Any],
    selected_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "phase1_posthoc_robustness",
        "source_phase1_run_id": selected["run_id"],
        "source_selected_layer_sha256": sha256_file(selected_path),
        "source_manifest_sha256": selected["manifest_sha256"],
        "config_sha256": config.identity_hash(),
        "seed": 42,
        "test_split_used": False,
        "k_values": list(K_VALUES),
        "pilot_pairs_per_task_split": PILOT_PAIRS_PER_TASK_SPLIT,
        "methods": list(METHODS),
        "packages": package_versions(("jspace-research", "jlens", "torch")),
    }


def run_metric_robustness(
    config: Phase1Config,
    selected_path: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, root, examples, activation_metadata, _ = _load_source(config, selected_path)
    layers = [int(value) for value in activation_metadata["layers"]]
    count = int(activation_metadata["number_examples"])
    width = int(activation_metadata["d_model"])
    score_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for layer in tqdm(layers, desc="Robustness metrics by layer"):
        reconstruction = _load_layer_reconstruction(
            root, selected, layer, count, width
        )
        scores = _scores_for_layer(root, layer, examples, reconstruction)
        training = scores[scores.split == "train"]
        threshold, training_ba = select_task_macro_balanced_threshold(
            training.label, training.score, training.task.tolist()
        )
        rows = _validation_rows(
            scores, layer=layer, threshold=threshold, tasks=config.tasks
        )
        for row in rows:
            row["training_macro_balanced_accuracy"] = training_ba
        metric_rows.extend(rows)
        score_frames.append(scores)
        del reconstruction

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(score_frames, ignore_index=True)
    ba_layer = _select_ba_layer(metrics)
    auprc_layer = int(selected["selected_layer"])
    metrics["selected_by_auprc"] = metrics.layer == auprc_layer
    metrics["selected_by_balanced_accuracy"] = metrics.layer == ba_layer
    atomic_write_csv(output_dir / "metric_layer_comparison.csv", metrics)
    _plot_metric_comparison(output_dir, metrics, auprc_layer, ba_layer)
    _plot_density_comparison(
        output_dir, scores, config.tasks, auprc_layer, ba_layer
    )
    update_provenance(
        output_dir / "provenance.json",
        _base_provenance(config, selected, selected_path),
        defaults={"metric_robustness_complete": False, "reconstruction_pilot_complete": False},
        updates={
            "metric_robustness_complete": True,
            "auprc_selected_layer": auprc_layer,
            "balanced_accuracy_comparison_layer": ba_layer,
            "run_layers": layers,
        },
    )
    print(f"Metric robustness complete: {output_dir / 'metric_layer_comparison.csv'}")
    return output_dir / "metric_layer_comparison.csv"


def _pilot_indices(examples: pd.DataFrame, tasks: tuple[str, ...]) -> np.ndarray:
    rng = np.random.default_rng(42)
    selected_indices: list[int] = []
    for split in ("train", "validation"):
        for task in tasks:
            subset = examples[(examples.split == split) & (examples.task == task)]
            pair_ids = np.asarray(sorted(subset.pair_id.unique().tolist()), dtype=object)
            count = min(PILOT_PAIRS_PER_TASK_SPLIT, len(pair_ids))
            chosen = set(rng.choice(pair_ids, size=count, replace=False).tolist())
            selected_indices.extend(subset[subset.pair_id.isin(chosen)].index.tolist())
    return np.asarray(sorted(selected_indices), dtype=np.int64)


def _pilot_layers(selected_layer: int, run_layers: list[int]) -> list[int]:
    layers = [layer for layer in range(selected_layer - 2, selected_layer + 3) if layer in run_layers]
    if not layers:
        raise RuntimeError("The Phase 1 cache does not include the selected layer")
    return layers


def _pilot_cache_paths(
    output_dir: Path, method: str, layer: int, sparsity_k: int
) -> tuple[Path, Path]:
    stem = output_dir / "cache" / method / f"layer_{layer:03d}_k{sparsity_k:03d}"
    return stem.with_suffix(".json"), stem.with_suffix(".pt")


def _load_or_compute_pilot(
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
    metadata_path, cache_path = _pilot_cache_paths(output_dir, method, layer, sparsity_k)
    identity = {
        "cache_schema_version": 1,
        "source_run_id": selected["run_id"],
        "source_manifest_sha256": selected["manifest_sha256"],
        "subset_sha256": subset_hash,
        "method": method,
        "layer": layer,
        "layer_position": layer_position,
        "sparsity_k": sparsity_k,
        "screen_candidates": config.screen_candidates if method == "screened_greedy" else None,
        "example_count": len(subset_indices),
        "width": int(activations.shape[2]),
    }
    if metadata_path.exists():
        if read_json(metadata_path) != identity:
            raise RuntimeError(f"Robustness cache identity mismatch: {metadata_path}")
        if not cache_path.is_file():
            raise RuntimeError(f"Robustness cache is incomplete: {cache_path}")
        value = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict) or value.get("example_indices").tolist() != subset_indices.tolist():
            raise RuntimeError(f"Robustness cache contents are invalid: {cache_path}")
        return value

    reconstructions: list[torch.Tensor] = []
    supports: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []
    description = f"{method} L{layer} k={sparsity_k}"
    batches = list(batched(subset_indices, config.decomposition_batch_size))
    for indices in tqdm(batches, desc=description):
        hidden = read_bfloat16_bits(
            np.asarray(activations[indices, layer_position, :]).copy()
        )
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


def run_reconstruction_pilot(
    config: Phase1Config,
    selected_path: Path,
    output_dir: Path,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("The reconstruction robustness pilot requires a CUDA GPU")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, root, examples, activation_metadata, activations = _load_source(
        config, selected_path
    )
    run_layers = [int(value) for value in activation_metadata["layers"]]
    pilot_layers = _pilot_layers(int(selected["selected_layer"]), run_layers)
    subset_indices = _pilot_indices(examples, config.tasks)
    subset_hash = hashlib.sha256(
        json.dumps(subset_indices.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    subset = examples.loc[subset_indices].copy()

    lens = JacobianLensAdapter.load(config)
    validate_lens_for_layers(lens, int(activation_metadata["d_model"]), pilot_layers)
    unembedding_path = _relative_artifact(
        root, selected["artifacts"]["activations"]["unembedding"]
    )
    unembedding = torch.load(unembedding_path, map_location="cpu", weights_only=True)
    if unembedding.ndim != 2 or unembedding.shape[1] != int(activation_metadata["d_model"]):
        raise RuntimeError("Phase 1 unembedding width does not match the activation cache")
    device = torch.device("cuda:0")
    cached: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
    for layer in pilot_layers:
        layer_position = run_layers.index(layer)
        dictionary = build_normalized_dictionary(
            jacobian=lens.jacobian(layer),
            unembedding=unembedding,
            layer=layer,
            device=device,
            chunk_size=config.dictionary_chunk_size,
        )
        for method in METHODS:
            for sparsity_k in K_VALUES:
                cached[(method, layer, sparsity_k)] = _load_or_compute_pilot(
                    output_dir=output_dir,
                    method=method,
                    layer=layer,
                    sparsity_k=sparsity_k,
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

    rows: list[dict[str, Any]] = []
    for layer in pilot_layers:
        layer_position = run_layers.index(layer)
        hidden = read_bfloat16_bits(
            np.asarray(activations[subset_indices, layer_position, :]).copy()
        )
        jaccard = _support_jaccard(
            cached[("screened_greedy", layer, 25)]["support_ids"],
            cached[("screened_greedy", layer, 25)]["coefficients"],
            cached[("gradient_pursuit", layer, 25)]["support_ids"],
            cached[("gradient_pursuit", layer, 25)]["coefficients"],
        )
        for method in METHODS:
            for sparsity_k in K_VALUES:
                value = cached[(method, layer, sparsity_k)]
                reconstruction = value["reconstruction"].float()
                quality = _reconstruction_quality(hidden, reconstruction)
                direction = learn_task_balanced_direction(
                    reconstruction[subset.split.to_numpy() == "train"],
                    subset.loc[subset.split == "train", "label"].to_numpy(),
                    subset.loc[subset.split == "train", "task"].tolist(),
                )
                scores = direction_scores(
                    reconstruction, direction["mu_clean"], direction["d_unit"]
                ).numpy()
                frame = subset[["split", "task", "label"]].copy()
                frame["score"] = scores
                training = frame[frame.split == "train"]
                threshold, training_ba = select_task_macro_balanced_threshold(
                    training.label, training.score, training.task.tolist()
                )
                result_rows = _validation_rows(
                    frame.reset_index(drop=True),
                    layer=layer,
                    threshold=threshold,
                    tasks=config.tasks,
                    method=method,
                    sparsity_k=sparsity_k,
                )
                for row in result_rows:
                    if row["scope"] == "task":
                        mask = subset.task.to_numpy() == row["task"]
                    else:
                        mask = np.ones(len(subset), dtype=bool)
                    train_mask = mask & (subset.split.to_numpy() == "train")
                    validation_mask = mask & (subset.split.to_numpy() == "validation")
                    for key, values in quality.items():
                        row[f"train_{key}"] = float(values[train_mask].mean())
                        row[f"validation_{key}"] = float(values[validation_mask].mean())
                    row["training_macro_balanced_accuracy"] = training_ba
                    row["support_jaccard_k25"] = (
                        float(jaccard[mask].mean()) if sparsity_k == 25 else np.nan
                    )
                rows.extend(result_rows)

    result = pd.DataFrame(rows)
    result_path = output_dir / "reconstruction_robustness.csv"
    atomic_write_csv(result_path, result)
    update_provenance(
        output_dir / "provenance.json",
        _base_provenance(config, selected, selected_path),
        defaults={"metric_robustness_complete": False, "reconstruction_pilot_complete": False},
        updates={
            "reconstruction_pilot_complete": True,
            "pilot_layers": pilot_layers,
            "pilot_example_indices_sha256": subset_hash,
            "pilot_example_count": len(subset_indices),
            "gradient_pursuit": {
                "dictionary_search": "full",
                "selection": "largest_positive_residual_correlation",
                "update": "active_support_gradient_with_exact_line_search",
                "nonnegativity": "step_clipped_to_feasible_coefficients",
            },
        },
    )
    print(f"Reconstruction robustness complete: {result_path}")
    return result_path


def run(
    config: Phase1Config,
    selected_path: str | Path,
    output_dir: str | Path,
    stage: str,
) -> None:
    selected = Path(selected_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == selected.parent or selected.parent in output.parents:
        raise ValueError("Robustness output must not be inside the frozen Phase 1 directory")
    if stage in ("metrics", "all"):
        run_metric_robustness(config, selected, output)
    if stage in ("reconstruction", "all"):
        run_reconstruction_pilot(config, selected, output)
    if stage not in ("metrics", "reconstruction", "all"):
        raise ValueError(f"Unknown robustness stage: {stage}")
