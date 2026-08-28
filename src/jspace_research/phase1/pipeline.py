from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..runtime import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    cuda_metadata,
    ensure_cache_metadata,
    package_versions,
    read_json,
    sha256_file,
    update_provenance,
)
from .adapters import (
    HuggingFaceModelAdapter,
    JacobianLensAdapter,
    load_tokenizer,
    validate_lens_for_layers,
    validate_model_lens,
)
from .cache import (
    load_done,
    open_memmap,
    open_uint16_memmap,
    save_done,
)
from .config import Phase1Config
from .data import (
    TASK_DISPLAY,
    expand_examples,
    manifest_sha256,
    prepare_manifest,
    read_jsonl,
    render_ids,
    validate_pair_manifest,
)
from .jspace import (
    batched,
    build_normalized_dictionary,
    compute_layer_metrics,
    direction_scores,
    learn_task_balanced_direction,
    read_bfloat16_bits,
    screened_nonnegative_pursuit,
    select_layer,
    tensor_to_bfloat16_bits,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ACTIVATION_CHECKPOINT_SIZE = 25


def _base_provenance(config: Phase1Config, manifest_digest: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": 1,
        "run_id": _run_id(config, manifest_digest),
        "resolved_config": config.scientific_dict(),
        "decomposition": {
            "method": "screened_nonnegative_greedy_approximation",
            "dictionary_l2_normalized": True,
            "sparsity_k": config.sparsity_k,
            "screen_candidates": config.screen_candidates,
        },
        "dtypes": {
            "model": config.model.precision,
            "activation_cache": "bfloat16",
            "reconstruction_cache": "bfloat16",
            "coefficient_cache": "float32",
            "support_id_cache": "int32",
        },
        "config_sha256": config.identity_hash(),
        "manifest_sha256": manifest_digest,
        "packages": package_versions(
            ("jspace-research", "jlens", "torch", "transformers")
        ),
    }


def _run_id(config: Phase1Config, manifest_digest: str) -> str:
    return f"phase1-{config.identity_hash()[:12]}-{manifest_digest[:12]}"


def _write_or_validate_provenance(
    config: Phase1Config,
    manifest_digest: str,
    *,
    updates: dict[str, Any] | None = None,
) -> None:
    update_provenance(
        config.output_dir / "provenance.json",
        _base_provenance(config, manifest_digest),
        defaults={
            "selected_layer": None,
            "run_layers": None,
            "gpu": None,
        },
        updates=updates,
    )


def _select_run_layers(source_layers: list[int], count: int | None) -> list[int]:
    if count is None or count >= len(source_layers):
        return list(source_layers)
    indices = np.linspace(0, len(source_layers) - 1, count).round().astype(int)
    return [source_layers[int(index)] for index in sorted(set(indices.tolist()))]


def _load_prepared(
    config: Phase1Config,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], str]:
    config.validate()
    manifest_path = config.output_dir / "pair_manifest.jsonl"
    if not manifest_path.exists():
        raise RuntimeError("Run the prepare stage before capture or analysis")
    rows = read_jsonl(manifest_path)
    validate_pair_manifest(rows, config)
    digest = manifest_sha256(manifest_path)
    _write_or_validate_provenance(config, digest)
    return manifest_path, rows, expand_examples(rows), digest


def prepare(config: Phase1Config) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config)
    manifest_path, rows = prepare_manifest(config, tokenizer)
    validate_pair_manifest(rows, config)
    digest = manifest_sha256(manifest_path)
    _write_or_validate_provenance(config, digest)
    print(f"Frozen manifest: {manifest_path} ({len(rows)} pairs)")
    return manifest_path


def capture(config: Phase1Config) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("The capture stage requires a CUDA GPU")
    _, _, examples, manifest_digest = _load_prepared(config)
    tokenizer = load_tokenizer(config)
    lens = JacobianLensAdapter.load(config)
    model = HuggingFaceModelAdapter.load(config, tokenizer)
    validate_model_lens(model, lens)

    run_layers = _select_run_layers(list(lens.source_layers), config.smoke_layer_count)
    number_examples = len(examples)
    width = model.hidden_width
    cache_identity = {
        "cache_schema_version": 2,
        "config_sha256": config.identity_hash(),
        "manifest_sha256": manifest_digest,
        "number_examples": number_examples,
        "layers": run_layers,
        "d_model": width,
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "lens_sha256": config.lens.sha256,
        "dtype": "bfloat16",
        "decision_point": "final_non_padding_prompt_token_before_generation",
    }
    cache_dir = config.output_dir / "cache"
    ensure_cache_metadata(cache_dir / "activations.json", cache_identity)
    activation_path = cache_dir / "activations_bf16.dat"
    done_path = cache_dir / "activations_done.npy"
    activations = open_uint16_memmap(activation_path, (number_examples, len(run_layers), width))
    done = load_done(done_path, number_examples)

    pending = np.flatnonzero(~done)
    with tqdm(total=len(pending), desc="Capture activations") as progress:
        for batch in batched(pending, ACTIVATION_CHECKPOINT_SIZE):
            for raw_index in batch:
                example_index = int(raw_index)
                input_ids = render_ids(tokenizer, examples[example_index]["messages"])
                if input_ids.shape[-1] > config.max_input_tokens:
                    raise RuntimeError(
                        f"Frozen example {example_index} exceeds max_input_tokens"
                    )
                stack = model.capture_final_prompt_token(input_ids, run_layers)
                activations[example_index] = tensor_to_bfloat16_bits(stack)
                progress.update(1)
            activations.flush()
            done[batch] = True
            save_done(done_path, done)

    unembedding_path = cache_dir / "unembedding_weight_bf16.pt"
    unembedding = model.unembedding()
    if unembedding_path.exists():
        existing = torch.load(unembedding_path, map_location="cpu", weights_only=True)
        if existing.shape != unembedding.shape or existing.dtype != unembedding.dtype:
            raise RuntimeError("Cached unembedding matrix is incompatible")
    else:
        torch.save(unembedding, unembedding_path)

    gpu = cuda_metadata(model_input_device=str(model.input_device))
    gpu["analysis_device"] = "cuda:0"
    _write_or_validate_provenance(
        config,
        manifest_digest,
        updates={
            "gpu": gpu,
            "run_layers": run_layers,
        },
    )
    del model, lens, unembedding, activations
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Activation cache complete: {activation_path}")
    return activation_path


def _read_cache_batch(
    memory_map: np.memmap, indices: np.ndarray, layer_position: int | None = None
) -> torch.Tensor:
    if layer_position is None:
        bits = memory_map[indices]
    else:
        bits = memory_map[indices, layer_position, :]
    return read_bfloat16_bits(bits)


def _decomposition_paths(output_dir: Path, layer: int) -> dict[str, Path]:
    layer_dir = output_dir / "cache" / "decompositions"
    stem = f"layer_{layer:03d}"
    return {
        "metadata": layer_dir / f"{stem}.json",
        "reconstruction": layer_dir / f"{stem}_bfloat16.dat",
        "support_ids": layer_dir / f"{stem}_support_ids_i32.dat",
        "coefficients": layer_dir / f"{stem}_coefficients_f32.dat",
        "done": layer_dir / f"{stem}_done.npy",
    }


def _decompose_layer(
    *,
    config: Phase1Config,
    lens: JacobianLensAdapter,
    unembedding: torch.Tensor,
    activations: np.memmap,
    layer: int,
    layer_position: int,
    cache_identity: dict[str, Any],
    device: torch.device,
) -> np.memmap:
    number_examples = activations.shape[0]
    width = activations.shape[2]
    paths = _decomposition_paths(config.output_dir, layer)
    metadata = {
        **cache_identity,
        "cache_schema_version": 2,
        "layer": layer,
        "layer_position": layer_position,
        "reconstruction_shape": [number_examples, width],
        "sparse_shape": [number_examples, config.sparsity_k],
        "reconstruction_dtype": "bfloat16",
        "support_id_dtype": "int32",
        "coefficient_dtype": "float32",
    }
    ensure_cache_metadata(paths["metadata"], metadata)
    decompositions = open_uint16_memmap(paths["reconstruction"], (number_examples, width))
    support_ids = open_memmap(
        paths["support_ids"],
        (number_examples, config.sparsity_k),
        dtype=np.int32,
        fill_value=-1,
    )
    coefficients = open_memmap(
        paths["coefficients"],
        (number_examples, config.sparsity_k),
        dtype=np.float32,
        fill_value=0.0,
    )
    done = load_done(paths["done"], number_examples)
    if bool(done.all()):
        del support_ids, coefficients
        return decompositions

    dictionary = build_normalized_dictionary(
        jacobian=lens.jacobian(layer),
        unembedding=unembedding,
        layer=layer,
        device=device,
        chunk_size=config.dictionary_chunk_size,
    )
    pending = np.where(~done)[0]
    for indices in tqdm(
        list(batched(pending, config.decomposition_batch_size)),
        desc=f"Decompose L{layer}",
    ):
        hidden = _read_cache_batch(activations, indices, layer_position)
        reconstructed, batch_support_ids, batch_coefficients = screened_nonnegative_pursuit(
            hidden,
            dictionary,
            sparsity_k=config.sparsity_k,
            screen_candidates=config.screen_candidates,
        )
        decompositions[indices] = tensor_to_bfloat16_bits(reconstructed)
        support_ids[indices] = batch_support_ids.numpy().astype(np.int32, copy=False)
        coefficients[indices] = batch_coefficients.numpy().astype(np.float32, copy=False)
        decompositions.flush()
        support_ids.flush()
        coefficients.flush()
        done[indices] = True
        save_done(paths["done"], done)
    del dictionary, support_ids, coefficients
    gc.collect()
    torch.cuda.empty_cache()
    return decompositions


def _analyze_layer(
    *,
    config: Phase1Config,
    layer: int,
    examples_frame: pd.DataFrame,
    decompositions: np.memmap,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_indices = examples_frame.index[examples_frame.split == "train"].to_numpy()
    validation_indices = examples_frame.index[examples_frame.split == "validation"].to_numpy()
    train_representations = _read_cache_batch(decompositions, train_indices)
    train_frame = examples_frame.loc[train_indices]
    artifact = learn_task_balanced_direction(
        train_representations,
        train_frame.label.to_numpy(),
        train_frame.task.tolist(),
    )
    artifact = {"layer": layer, **artifact}
    artifact_dir = config.output_dir / "layer_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_dir / f"layer_{layer:03d}.pt")
    del train_representations

    records: list[dict[str, Any]] = []
    for indices in batched(validation_indices, config.decomposition_batch_size * 4):
        representations = _read_cache_batch(decompositions, indices)
        scores = direction_scores(
            representations,
            artifact["mu_clean"],
            artifact["d_unit"],
        )
        for offset, index in enumerate(indices.tolist()):
            row = examples_frame.loc[index]
            records.append(
                {
                    "layer": layer,
                    "example_index": int(row.example_index),
                    "pair_id": row.pair_id,
                    "task": row.task,
                    "task_display": row.task_display,
                    "condition": row.condition,
                    "label": int(row.label),
                    "score": float(scores[offset]),
                }
            )
    scores = pd.DataFrame(records)
    score_dir = config.output_dir / "layer_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(score_dir / f"layer_{layer:03d}.parquet", index=False)
    return scores, artifact


def _save_plots(
    config: Phase1Config,
    metrics: pd.DataFrame,
    validation_scores: pd.DataFrame,
    selected_layer: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    macro = metrics[metrics.scope == "macro"].sort_values("layer")
    axis.plot(macro.layer, macro.auprc, marker="o", label="Macro AUPRC")
    for task in config.tasks:
        subset = metrics[(metrics.scope == "task") & (metrics.task == task)].sort_values("layer")
        axis.plot(subset.layer, subset.auprc, alpha=0.65, label=TASK_DISPLAY[task])
    axis.axvline(selected_layer, linestyle="--", label=f"Selected L{selected_layer}")
    axis.set_xlabel("J-lens layer")
    axis.set_ylabel("Validation AUPRC")
    axis.set_title("J-Space Prompt-Injection Detection by Layer")
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.output_dir / "layer_auprc.png", dpi=180)
    plt.close(figure)

    selected = validation_scores[validation_scores.layer == selected_layer]
    figure, axis = plt.subplots(figsize=(8, 5))
    for condition in ("control", "attack"):
        values = selected[selected.condition == condition].score.to_numpy()
        axis.hist(values, bins=40, alpha=0.5, density=True, label=condition)
    axis.set_xlabel("Clean-to-attack direction score")
    axis.set_ylabel("Density")
    axis.set_title(f"Validation Scores at Selected Layer {selected_layer}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.output_dir / "selected_layer_score_distribution.png", dpi=180)
    plt.close(figure)


def _build_selected_result(
    *,
    config: Phase1Config,
    manifest_digest: str,
    run_layers: list[int],
    selected_layer: int,
    selection_value: float,
    macro_auroc: float,
    direction_norm: float,
    direction_path: Path,
) -> dict[str, Any]:
    selected_layer_position = run_layers.index(selected_layer)
    selected_cache_paths = _decomposition_paths(config.output_dir, selected_layer)

    def relative(path: Path) -> str:
        return str(path.relative_to(config.output_dir))

    direction_sha256 = sha256_file(direction_path)
    return {
        "schema_version": 1,
        "phase": 1,
        "run_id": _run_id(config, manifest_digest),
        "frozen": True,
        "selected_layer": selected_layer,
        "selected_layer_position": selected_layer_position,
        "selection_metric": "macro_task_auprc",
        "selection_value": selection_value,
        "macro_auroc": macro_auroc,
        "direction_norm": direction_norm,
        "direction_artifact": direction_path.name,
        "resolved_config": config.scientific_dict(),
        "config_sha256": config.identity_hash(),
        "manifest_sha256": manifest_digest,
        "decomposition": {
            "method": "screened_nonnegative_greedy_approximation",
            "dictionary_l2_normalized": True,
            "sparsity_k": config.sparsity_k,
            "screen_candidates": config.screen_candidates,
        },
        "artifacts": {
            "provenance": "provenance.json",
            "pair_manifest": "pair_manifest.jsonl",
            "direction": {
                "path": direction_path.name,
                "sha256": direction_sha256,
            },
            "activations": {
                "metadata": "cache/activations.json",
                "residuals": "cache/activations_bf16.dat",
                "completion": "cache/activations_done.npy",
                "unembedding": "cache/unembedding_weight_bf16.pt",
                "layer_position": selected_layer_position,
            },
            "selected_layer_decomposition": {
                "metadata": relative(selected_cache_paths["metadata"]),
                "reconstruction": relative(selected_cache_paths["reconstruction"]),
                "support_ids": relative(selected_cache_paths["support_ids"]),
                "coefficients": relative(selected_cache_paths["coefficients"]),
                "completion": relative(selected_cache_paths["done"]),
            },
            "metrics": "layer_metrics.csv",
            "validation_scores": "validation_scores.parquet",
        },
    }


def analyze(config: Phase1Config) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("The analyze stage requires a CUDA GPU")
    _, _, examples, manifest_digest = _load_prepared(config)
    lens = JacobianLensAdapter.load(config)
    cache_dir = config.output_dir / "cache"
    activation_metadata = read_json(cache_dir / "activations.json")
    if (
        activation_metadata["config_sha256"] != config.identity_hash()
        or activation_metadata["manifest_sha256"] != manifest_digest
    ):
        raise RuntimeError("Activation cache identity does not match this run")
    run_layers = [int(layer) for layer in activation_metadata["layers"]]
    number_examples = int(activation_metadata["number_examples"])
    width = int(activation_metadata["d_model"])
    validate_lens_for_layers(lens, width, run_layers)
    activation_done = load_done(cache_dir / "activations_done.npy", number_examples)
    if not bool(activation_done.all()):
        raise RuntimeError("Activation capture is incomplete")
    activations = open_uint16_memmap(
        cache_dir / "activations_bf16.dat",
        (number_examples, len(run_layers), width),
    )
    unembedding = torch.load(
        cache_dir / "unembedding_weight_bf16.pt",
        map_location="cpu",
        weights_only=True,
    )
    if tuple(unembedding.shape) != (unembedding.shape[0], width):
        raise RuntimeError("Unembedding width does not match activation width")

    examples_frame = pd.DataFrame(examples).set_index("example_index", drop=False)
    device = torch.device("cuda:0")
    cache_identity = {
        "config_sha256": config.identity_hash(),
        "manifest_sha256": manifest_digest,
        "lens_sha256": config.lens.sha256,
        "sparsity_k": config.sparsity_k,
        "screen_candidates": config.screen_candidates,
    }
    score_frames: list[pd.DataFrame] = []
    artifacts: dict[int, dict[str, Any]] = {}
    for layer_position, layer in enumerate(run_layers):
        decompositions = _decompose_layer(
            config=config,
            lens=lens,
            unembedding=unembedding,
            activations=activations,
            layer=layer,
            layer_position=layer_position,
            cache_identity=cache_identity,
            device=device,
        )
        scores, artifact = _analyze_layer(
            config=config,
            layer=layer,
            examples_frame=examples_frame,
            decompositions=decompositions,
        )
        score_frames.append(scores)
        artifacts[layer] = artifact
        del decompositions

    validation_scores = pd.concat(score_frames, ignore_index=True)
    atomic_write_parquet(
        config.output_dir / "validation_scores.parquet", validation_scores
    )
    metrics = compute_layer_metrics(validation_scores, run_layers, config.tasks, TASK_DISPLAY)
    atomic_write_csv(config.output_dir / "layer_metrics.csv", metrics)
    best = select_layer(metrics)
    selected_layer = int(best.layer)
    selected_artifact = artifacts[selected_layer]
    selected_tensor_path = config.output_dir / "selected_layer_direction.pt"
    torch.save(
        {
            "layer": selected_layer,
            "mu_clean": selected_artifact["mu_clean"],
            "d_raw": selected_artifact["d_raw"],
            "d_norm": selected_artifact["d_norm"],
            "d_unit": selected_artifact["d_unit"],
        },
        selected_tensor_path,
    )
    selected_result = _build_selected_result(
        config=config,
        manifest_digest=manifest_digest,
        run_layers=run_layers,
        selected_layer=selected_layer,
        selection_value=float(best.auprc),
        macro_auroc=float(best.auroc),
        direction_norm=float(selected_artifact["d_norm"]),
        direction_path=selected_tensor_path,
    )
    direction_sha256 = selected_result["artifacts"]["direction"]["sha256"]
    selected_path = config.output_dir / "selected_layer.json"
    atomic_write_json(selected_path, selected_result)
    _save_plots(config, metrics, validation_scores, selected_layer)
    _write_or_validate_provenance(
        config,
        manifest_digest,
        updates={
            "run_layers": run_layers,
            "selected_layer": selected_layer,
            "selected_direction_sha256": direction_sha256,
        },
    )
    print(json.dumps(selected_result, indent=2))
    return selected_path


def run(config: Phase1Config, stage: str) -> None:
    if stage == "prepare":
        prepare(config)
    elif stage == "capture":
        capture(config)
    elif stage == "analyze":
        analyze(config)
    elif stage == "all":
        prepare(config)
        capture(config)
        analyze(config)
    else:
        raise ValueError(f"Unknown Phase 1 stage: {stage}")
