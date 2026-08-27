from __future__ import annotations

import gc
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .cache import (
    atomic_write_json,
    ensure_cache_metadata,
    load_done,
    open_uint16_memmap,
    read_json,
    save_done,
    sha256_file,
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


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _base_provenance(config: Phase1Config, manifest_digest: str) -> dict[str, Any]:
    return {
        "model": {
            "id": config.model.id,
            "revision": config.model.revision,
            "precision": config.model.precision,
        },
        "lens": {
            "repository": config.lens.repository,
            "revision": config.lens.revision,
            "filename": config.lens.filename,
            "sha256": config.lens.sha256,
        },
        "dependencies": {
            "jacobian_lens_revision": config.dependencies.jacobian_lens_revision,
            "bipia_revision": config.dependencies.bipia_revision,
        },
        "experiment": {
            "seed": config.seed,
            "tasks": list(config.tasks),
            "train_pairs_per_task": config.train_pairs_per_task,
            "validation_pairs_per_task": config.validation_pairs_per_task,
            "max_input_tokens": config.max_input_tokens,
            "sparsity_k": config.sparsity_k,
            "screen_candidates": config.screen_candidates,
        },
        "config_sha256": config.identity_hash(),
        "manifest_sha256": manifest_digest,
        "packages": {
            name: _package_version(name)
            for name in ("jspace-research", "jlens", "torch", "transformers")
        },
    }


def _write_or_validate_provenance(
    config: Phase1Config,
    manifest_digest: str,
    *,
    updates: dict[str, Any] | None = None,
) -> None:
    path = config.output_dir / "provenance.json"
    base = _base_provenance(config, manifest_digest)
    if path.exists():
        current = read_json(path)
        for key, expected in base.items():
            if current.get(key) != expected:
                raise RuntimeError(f"Provenance mismatch at {path}; use a new output directory")
        value = current
    else:
        value = base
    if updates:
        value.update(updates)
    atomic_write_json(path, value)


def _load_tokenizer(config: Phase1Config) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.model.id, revision=config.model.revision)


def _load_lens(config: Phase1Config) -> tuple[Any, Path]:
    from huggingface_hub import hf_hub_download
    from jlens import JacobianLens

    path = Path(
        hf_hub_download(
            repo_id=config.lens.repository,
            filename=config.lens.filename,
            revision=config.lens.revision,
        )
    )
    actual = sha256_file(path)
    if actual != config.lens.sha256:
        raise RuntimeError(f"Lens SHA-256 mismatch: expected {config.lens.sha256}, found {actual}")
    return JacobianLens.load(str(path)), path


def _select_run_layers(source_layers: list[int], count: int | None) -> list[int]:
    if count is None or count >= len(source_layers):
        return list(source_layers)
    indices = np.linspace(0, len(source_layers) - 1, count).round().astype(int)
    return [source_layers[int(index)] for index in sorted(set(indices.tolist()))]


def _load_prepared(
    config: Phase1Config,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], str]:
    config.validate(require_data_files=True)
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
    tokenizer = _load_tokenizer(config)
    manifest_path, rows = prepare_manifest(config, tokenizer)
    validate_pair_manifest(rows, config)
    digest = manifest_sha256(manifest_path)
    _write_or_validate_provenance(config, digest)
    print(f"Frozen manifest: {manifest_path} ({len(rows)} pairs)")
    return manifest_path


def _load_model(config: Phase1Config) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs = {
        "revision": config.model.revision,
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(config.model.id, **kwargs)
    except Exception as causal_error:
        try:
            from transformers import AutoModelForMultimodalLM

            print(
                "AutoModelForCausalLM failed; trying AutoModelForMultimodalLM: "
                f"{type(causal_error).__name__}"
            )
            return AutoModelForMultimodalLM.from_pretrained(config.model.id, **kwargs)
        except Exception:
            raise causal_error from None


def capture(config: Phase1Config) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("The capture stage requires a CUDA GPU")
    _, _, examples, manifest_digest = _load_prepared(config)
    tokenizer = _load_tokenizer(config)
    lens, _ = _load_lens(config)

    import jlens

    hf_model = _load_model(config)
    hf_model.eval()
    lens_model = jlens.from_hf(hf_model, tokenizer, compile=False)
    if lens_model.d_model != lens.d_model:
        raise RuntimeError(f"Model/lens width mismatch: {lens_model.d_model} != {lens.d_model}")
    if max(lens.source_layers) >= lens_model.n_layers:
        raise RuntimeError("The fitted lens contains an out-of-range source layer")

    run_layers = _select_run_layers(list(lens.source_layers), config.smoke_layer_count)
    number_examples = len(examples)
    width = lens_model.d_model
    cache_identity = {
        "config_sha256": config.identity_hash(),
        "manifest_sha256": manifest_digest,
        "number_examples": number_examples,
        "layers": run_layers,
        "d_model": width,
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "lens_sha256": config.lens.sha256,
    }
    cache_dir = config.output_dir / "cache"
    ensure_cache_metadata(cache_dir / "activations.json", cache_identity)
    activation_path = cache_dir / "activations_bf16.dat"
    done_path = cache_dir / "activations_done.npy"
    activations = open_uint16_memmap(activation_path, (number_examples, len(run_layers), width))
    done = load_done(done_path, number_examples)

    for example_index in tqdm(range(number_examples), desc="Capture activations"):
        if done[example_index]:
            continue
        input_ids = render_ids(tokenizer, examples[example_index]["messages"])
        if input_ids.shape[-1] > config.max_input_tokens:
            raise RuntimeError(f"Frozen example {example_index} exceeds max_input_tokens")
        input_ids = input_ids.to(lens_model.input_device)
        with (
            torch.no_grad(),
            jlens.ActivationRecorder(lens_model.layers, at=run_layers) as recorder,
        ):
            lens_model.forward(input_ids)
        if set(recorder.activations) != set(run_layers):
            raise RuntimeError(f"Missing captured layers for example {example_index}")
        stack = torch.stack(
            [recorder.activations[layer][0, -1, :].detach() for layer in run_layers]
        )
        activations[example_index] = tensor_to_bfloat16_bits(stack)
        activations.flush()
        done[example_index] = True
        save_done(done_path, done)

    unembedding_path = cache_dir / "unembedding_weight_bf16.pt"
    unembedding = lens_model._lm_head.weight.detach().to("cpu", dtype=torch.bfloat16).contiguous()
    if unembedding_path.exists():
        existing = torch.load(unembedding_path, map_location="cpu", weights_only=True)
        if existing.shape != unembedding.shape or existing.dtype != unembedding.dtype:
            raise RuntimeError("Cached unembedding matrix is incompatible")
    else:
        torch.save(unembedding, unembedding_path)

    gpu = torch.cuda.get_device_properties(0)
    _write_or_validate_provenance(
        config,
        manifest_digest,
        updates={
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(gpu.total_memory),
            },
            "run_layers": run_layers,
        },
    )
    del lens_model, hf_model, unembedding, activations
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


def _decompose_layer(
    *,
    config: Phase1Config,
    lens: Any,
    unembedding: torch.Tensor,
    activations: np.memmap,
    layer: int,
    layer_position: int,
    cache_identity: dict[str, Any],
    device: torch.device,
) -> np.memmap:
    number_examples = activations.shape[0]
    width = activations.shape[2]
    layer_dir = config.output_dir / "cache" / "decompositions"
    layer_path = layer_dir / f"layer_{layer:03d}_bfloat16.dat"
    done_path = layer_dir / f"layer_{layer:03d}_done.npy"
    metadata = {
        **cache_identity,
        "layer": layer,
        "layer_position": layer_position,
        "shape": [number_examples, width],
    }
    ensure_cache_metadata(layer_dir / f"layer_{layer:03d}.json", metadata)
    decompositions = open_uint16_memmap(layer_path, (number_examples, width))
    done = load_done(done_path, number_examples)
    if bool(done.all()):
        return decompositions

    dictionary = build_normalized_dictionary(
        lens=lens,
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
        reconstructed, _, _ = screened_nonnegative_pursuit(
            hidden,
            dictionary,
            sparsity_k=config.sparsity_k,
            screen_candidates=config.screen_candidates,
        )
        decompositions[indices] = tensor_to_bfloat16_bits(reconstructed)
        decompositions.flush()
        done[indices] = True
        save_done(done_path, done)
    del dictionary
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


def analyze(config: Phase1Config) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("The analyze stage requires a CUDA GPU")
    _, _, examples, manifest_digest = _load_prepared(config)
    lens, _ = _load_lens(config)
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
    validation_scores.to_parquet(config.output_dir / "validation_scores.parquet", index=False)
    metrics = compute_layer_metrics(validation_scores, run_layers, config.tasks, TASK_DISPLAY)
    metrics.to_csv(config.output_dir / "layer_metrics.csv", index=False)
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
    selected_result = {
        "selected_layer": selected_layer,
        "selection_metric": "macro_task_auprc",
        "selection_value": float(best.auprc),
        "macro_auroc": float(best.auroc),
        "direction_artifact": selected_tensor_path.name,
    }
    selected_path = config.output_dir / "selected_layer.json"
    atomic_write_json(selected_path, selected_result)
    _save_plots(config, metrics, validation_scores, selected_layer)
    _write_or_validate_provenance(config, manifest_digest, updates={"run_layers": run_layers})
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
