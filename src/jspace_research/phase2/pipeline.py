from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..model import HuggingFaceModelAdapter, load_tokenizer
from ..phase1.cache import atomic_write_json, ensure_cache_metadata, read_json, sha256_file
from ..phase1.data import render_ids
from .artifacts import Phase1Handoff, load_phase1_handoff
from .config import Phase2Config
from .scoring import (
    JUDGE_RUBRIC_SHA256,
    OpenAIAttackJudge,
    qualitative_examples,
    score_generation,
    summarize_results,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _package_versions() -> dict[str, str | None]:
    return {
        name: _package_version(name)
        for name in (
            "jspace-research",
            "jlens",
            "openai",
            "pandas",
            "rouge-score",
            "torch",
            "transformers",
        )
    }


def _gpu_devices() -> list[dict[str, Any]]:
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return devices


def _run_id(config: Phase2Config, handoff: Phase1Handoff) -> str:
    return f"phase2-{config.identity_hash()[:12]}-{handoff.metadata['manifest_sha256'][:12]}"


def _base_provenance(config: Phase2Config, handoff: Phase1Handoff) -> dict[str, Any]:
    phase1_settings = config.phase1.scientific_dict()
    return {
        "schema_version": 1,
        "phase": 2,
        "run_id": _run_id(config, handoff),
        "phase1_run_id": handoff.metadata["run_id"],
        "phase1_config_sha256": handoff.metadata["config_sha256"],
        "manifest_sha256": handoff.metadata["manifest_sha256"],
        "selected_layer": handoff.selected_layer,
        "decomposition": handoff.metadata["decomposition"],
        "model": phase1_settings["model"],
        "lens": phase1_settings["lens"],
        "dependencies": phase1_settings["dependencies"],
        "seed": config.phase1.seed,
        "dtype": config.phase1.model.precision,
        "resolved_config": config.scientific_dict(),
        "config_sha256": config.identity_hash(),
        "judge": {
            "model": config.judge_model,
            "rubric_sha256": JUDGE_RUBRIC_SHA256,
            "labels": ["YES", "NO", "UNKNOWN"],
            "successful_label": "YES",
        },
    }


def _write_or_validate_provenance(
    config: Phase2Config,
    handoff: Phase1Handoff,
    *,
    updates: dict[str, Any] | None = None,
) -> None:
    path = config.output_dir / "provenance.json"
    base = _base_provenance(config, handoff)
    if path.exists():
        current = read_json(path)
        for key, expected in base.items():
            if current.get(key) != expected:
                raise RuntimeError(f"Provenance mismatch at {path}; use a new output directory")
        value = current
    else:
        value = {
            **base,
            "generation_gpu": None,
            "generation_packages": None,
            "analysis_packages": None,
        }
    if updates:
        value.update(updates)
    atomic_write_json(path, value)


def _generation_dir(config: Phase2Config) -> Path:
    return config.output_dir / "cache" / "generations"


def _judgment_dir(config: Phase2Config) -> Path:
    return config.output_dir / "cache" / "judgments"


def _job_id(example_index: int, alpha_index: int) -> str:
    return f"example_{example_index:06d}_alpha_{alpha_index}"


def _generation_identity(config: Phase2Config, handoff: Phase1Handoff) -> dict[str, Any]:
    return {
        "cache_schema_version": 1,
        "phase2_config_sha256": config.identity_hash(),
        "phase1_run_id": handoff.metadata["run_id"],
        "manifest_sha256": handoff.metadata["manifest_sha256"],
        "model_id": config.phase1.model.id,
        "model_revision": config.phase1.model.revision,
        "selected_layer": handoff.selected_layer,
        "alphas": list(config.alphas),
        "max_new_tokens": config.max_new_tokens,
        "number_validation_examples": len(handoff.validation_examples),
        "number_jobs": len(handoff.validation_examples) * len(config.alphas),
        "decision_point": "final_non_padding_prompt_token_before_generation",
    }


def _expected_generation_fields(
    config: Phase2Config,
    handoff: Phase1Handoff,
    example: dict[str, Any],
    alpha_index: int,
) -> dict[str, Any]:
    alpha = config.alphas[alpha_index]
    return {
        "schema_version": 1,
        "job_id": _job_id(int(example["example_index"]), alpha_index),
        "phase2_config_sha256": config.identity_hash(),
        "phase1_run_id": handoff.metadata["run_id"],
        "example_id": f"{example['pair_id']}:{example['condition']}",
        "example_index": int(example["example_index"]),
        "pair_id": example["pair_id"],
        "task": example["task"],
        "condition": example["condition"],
        "prompt_hash": example["prompt_hash"],
        "alpha": alpha,
        "alpha_index": alpha_index,
    }


def _validate_cached_fields(path: Path, value: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"Cached result identity mismatch at {path}; use a new output directory")


def generate(config: Phase2Config) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("The Phase 2 generate stage requires a CUDA GPU")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    handoff = load_phase1_handoff(config)
    _write_or_validate_provenance(config, handoff)
    generation_dir = _generation_dir(config)
    ensure_cache_metadata(generation_dir / "metadata.json", _generation_identity(config, handoff))

    expected_paths = [
        generation_dir
        / f"{_job_id(int(example['example_index']), alpha_index)}.json"
        for example in handoff.validation_examples
        for alpha_index in range(len(config.alphas))
    ]
    if all(path.exists() for path in expected_paths):
        _load_generation_records(config, handoff)
        provenance = read_json(config.output_dir / "provenance.json")
        updates: dict[str, Any] = {}
        if provenance.get("generation_gpu") is None:
            devices = _gpu_devices()
            updates["generation_gpu"] = {
                "device_count": len(devices),
                "devices": devices,
                "model_input_device": None,
            }
        if provenance.get("generation_packages") is None:
            updates["generation_packages"] = _package_versions()
        if updates:
            _write_or_validate_provenance(config, handoff, updates=updates)
        print(f"Phase 2 generation cache already complete: {generation_dir}")
        return generation_dir

    tokenizer = load_tokenizer(config.phase1)
    model = HuggingFaceModelAdapter.load(config.phase1, tokenizer)
    if model.hidden_width != handoff.reconstruction_shape[1]:
        raise RuntimeError("Model width does not match the selected-layer reconstruction cache")

    for example in tqdm(handoff.validation_examples, desc="Phase 2 intervention generation"):
        example_index = int(example["example_index"])
        input_ids = render_ids(tokenizer, example["messages"])
        if int(input_ids.shape[-1]) > config.phase1.max_input_tokens:
            raise RuntimeError(f"Frozen example {example_index} exceeds max_input_tokens")
        reconstructed = handoff.reconstructed_jspace(example_index)

        for alpha_index, alpha in enumerate(config.alphas):
            expected = _expected_generation_fields(
                config, handoff, example, alpha_index
            )
            path = generation_dir / f"{expected['job_id']}.json"
            if path.exists():
                cached = read_json(path)
                _validate_cached_fields(path, cached, expected)
                if config.smoke and alpha == 0.0 and cached.get("zero_hook_equivalent") is not True:
                    raise RuntimeError(f"Smoke zero-hook equivalence is missing at {path}")
                continue

            generated_ids = model.generate_from_prompt(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                layer=handoff.selected_layer,
                reconstructed_jspace=reconstructed,
                alpha=alpha,
            )
            zero_hook_equivalent: bool | None = None
            if config.smoke and alpha == 0.0:
                no_hook_ids = model.generate_from_prompt(
                    input_ids,
                    max_new_tokens=config.max_new_tokens,
                )
                zero_hook_equivalent = bool(torch.equal(generated_ids, no_hook_ids))
                if not zero_hook_equivalent:
                    raise RuntimeError(
                        f"No-hook and alpha=0 hooked generation differ for example {example_index}"
                    )
            token_ids = [int(value) for value in generated_ids.tolist()]
            generation = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            atomic_write_json(
                path,
                {
                    **expected,
                    "generated_token_ids": token_ids,
                    "generation": generation,
                    "generation_sha256": _text_sha256(generation),
                    "zero_hook_equivalent": zero_hook_equivalent,
                },
            )

    gpu_devices = _gpu_devices()
    _write_or_validate_provenance(
        config,
        handoff,
        updates={
            "generation_gpu": {
                "device_count": len(gpu_devices),
                "devices": gpu_devices,
                "model_input_device": str(model.input_device),
            },
            "generation_packages": _package_versions(),
        },
    )
    del model, tokenizer, handoff
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Phase 2 generation cache complete: {generation_dir}")
    return generation_dir


def _load_generation_records(
    config: Phase2Config, handoff: Phase1Handoff
) -> list[dict[str, Any]]:
    generation_dir = _generation_dir(config)
    ensure_cache_metadata(generation_dir / "metadata.json", _generation_identity(config, handoff))
    records: list[dict[str, Any]] = []
    for example in handoff.validation_examples:
        for alpha_index in range(len(config.alphas)):
            expected = _expected_generation_fields(config, handoff, example, alpha_index)
            path = generation_dir / f"{expected['job_id']}.json"
            if not path.exists():
                raise RuntimeError(f"Phase 2 generation is incomplete; missing {path}")
            value = read_json(path)
            _validate_cached_fields(path, value, expected)
            if not isinstance(value.get("generated_token_ids"), list) or not isinstance(
                value.get("generation"), str
            ):
                raise RuntimeError(f"Generation cache is incomplete at {path}")
            if value.get("generation_sha256") != _text_sha256(value["generation"]):
                raise RuntimeError(f"Generation hash mismatch at {path}")
            if config.smoke and value["alpha"] == 0.0 and value.get(
                "zero_hook_equivalent"
            ) is not True:
                raise RuntimeError(f"Smoke zero-hook equivalence is missing at {path}")
            records.append({**example, **value})
    return records


def _judgment_identity(config: Phase2Config, handoff: Phase1Handoff) -> dict[str, Any]:
    return {
        "cache_schema_version": 1,
        "phase2_config_sha256": config.identity_hash(),
        "phase1_run_id": handoff.metadata["run_id"],
        "manifest_sha256": handoff.metadata["manifest_sha256"],
        "judge_model": config.judge_model,
        "judge_rubric_sha256": JUDGE_RUBRIC_SHA256,
        "labels": ["YES", "NO", "UNKNOWN"],
    }


def _load_or_create_judgment(
    *,
    config: Phase2Config,
    record: dict[str, Any],
    judge: Any,
) -> dict[str, Any]:
    judgment_dir = _judgment_dir(config)
    path = judgment_dir / f"{record['job_id']}.json"
    expected = {
        "schema_version": 1,
        "job_id": record["job_id"],
        "phase2_config_sha256": config.identity_hash(),
        "judge_model": config.judge_model,
        "judge_rubric_sha256": JUDGE_RUBRIC_SHA256,
        "generation_sha256": record["generation_sha256"],
        "attack_text_sha256": _text_sha256(record["attack_text"]),
    }
    if path.exists():
        value = read_json(path)
        _validate_cached_fields(path, value, expected)
        if value.get("judge_label") not in {"YES", "NO", "UNKNOWN"}:
            raise RuntimeError(f"Judgment cache is incomplete at {path}")
        return value
    label = judge.judge(record["attack_text"], record["generation"])
    if label not in {"YES", "NO", "UNKNOWN"}:
        raise RuntimeError(f"Judge returned an invalid label: {label!r}")
    value = {
        **expected,
        "judge_label": label,
        "attack_success": label == "YES",
    }
    atomic_write_json(path, value)
    return value


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_plots(config: Phase2Config, summary: pd.DataFrame) -> None:
    asr = summary[
        (summary.metric == "asr")
        & (summary.scope == "overall")
        & (summary.condition == "attack")
    ].sort_values("alpha")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(asr.alpha, asr.value, marker="o")
    axis.set_xlabel("J-space removal strength (alpha)")
    axis.set_ylabel("Attack Success Rate")
    axis.set_title("Phase 2 Attack Success vs J-Space Removal")
    axis.set_xticks(list(config.alphas))
    figure.tight_layout()
    figure.savefig(config.output_dir / "phase2_asr_vs_alpha.png", dpi=180)
    plt.close(figure)

    utility = summary[
        (summary.metric == "rougeL_recall")
        & (summary.scope == "task")
        & (summary.condition == "control")
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    for task in sorted(utility.task.dropna().unique()):
        task_rows = utility[utility.task == task].sort_values("alpha")
        axis.plot(task_rows.alpha, task_rows.value, marker="o", label=task)
    axis.set_xlabel("J-space removal strength (alpha)")
    axis.set_ylabel("ROUGE-L recall")
    axis.set_title("Clean Reference-Overlap Utility vs J-Space Removal")
    axis.set_xticks(list(config.alphas))
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.output_dir / "phase2_clean_utility_vs_alpha.png", dpi=180)
    plt.close(figure)


def analyze(config: Phase2Config, *, judge: Any | None = None) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    handoff = load_phase1_handoff(config)
    _write_or_validate_provenance(config, handoff)
    records = _load_generation_records(config, handoff)
    judgment_dir = _judgment_dir(config)
    ensure_cache_metadata(judgment_dir / "metadata.json", _judgment_identity(config, handoff))

    result_rows: list[dict[str, Any]] = []
    active_judge = judge
    for record in tqdm(records, desc="Phase 2 scoring"):
        attack_success: bool | None = None
        judge_label: str | None = None
        if record["condition"] == "attack":
            judgment_path = judgment_dir / f"{record['job_id']}.json"
            if active_judge is None and not judgment_path.exists():
                active_judge = OpenAIAttackJudge(config.judge_model)
            judgment = _load_or_create_judgment(
                config=config,
                record=record,
                judge=active_judge,
            )
            attack_success = bool(judgment["attack_success"])
            judge_label = str(judgment["judge_label"])
        scores = score_generation(record["generation"], record["target"])
        result_rows.append(
            {
                "example_id": record["example_id"],
                "example_index": int(record["example_index"]),
                "pair_id": record["pair_id"],
                "task": record["task"],
                "task_display": record["task_display"],
                "condition": record["condition"],
                "attack_category": record["attack_category"],
                "attack_variant_id": int(record["attack_variant_id"]),
                "position": record["position"],
                "alpha": float(record["alpha"]),
                "generation": record["generation"],
                "attack_success": attack_success,
                "judge_label": judge_label,
                **scores,
            }
        )

    results = pd.DataFrame(result_rows)
    baselines = results[results.alpha == 0.0].set_index("example_id")["generation"]
    results["baseline_generation"] = results.example_id.map(baselines)
    results["attack_success"] = pd.array(results.attack_success, dtype="boolean")
    results["is_valid"] = pd.array(results.is_valid, dtype="boolean")
    results["malformed"] = pd.array(results.malformed, dtype="boolean")
    result_columns = [
        "example_id",
        "example_index",
        "pair_id",
        "task",
        "task_display",
        "condition",
        "attack_category",
        "attack_variant_id",
        "position",
        "alpha",
        "baseline_generation",
        "generation",
        "attack_success",
        "judge_label",
        "task_score_name",
        "task_score",
        "rouge1_recall",
        "rouge2_recall",
        "rougeL_recall",
        "rougeLsum_recall",
        "refusal",
        "validity_defined",
        "is_valid",
        "malformed",
    ]
    results = results[result_columns].sort_values(["example_index", "alpha"])
    results_path = config.output_dir / "phase2_results.parquet"
    _atomic_write_parquet(results_path, results)

    summary = summarize_results(results)
    summary_path = config.output_dir / "phase2_summary.csv"
    _atomic_write_csv(summary_path, summary)
    examples_path = config.output_dir / "phase2_examples.csv"
    _atomic_write_csv(examples_path, qualitative_examples(results))
    _save_plots(config, summary)
    existing_analysis_packages = read_json(config.output_dir / "provenance.json").get(
        "analysis_packages"
    )
    _write_or_validate_provenance(
        config,
        handoff,
        updates={
            "artifacts": {
                "results": {
                    "path": results_path.name,
                    "sha256": sha256_file(results_path),
                },
                "summary": {
                    "path": summary_path.name,
                    "sha256": sha256_file(summary_path),
                },
                "examples": {
                    "path": examples_path.name,
                    "sha256": sha256_file(examples_path),
                },
                "asr_plot": "phase2_asr_vs_alpha.png",
                "utility_plot": "phase2_clean_utility_vs_alpha.png",
            },
            "analysis_packages": existing_analysis_packages or _package_versions(),
        },
    )
    print(f"Phase 2 results: {results_path}")
    return results_path


def run(config: Phase2Config, stage: str) -> None:
    if stage == "generate":
        generate(config)
    elif stage == "analyze":
        analyze(config)
    elif stage == "all":
        generate(config)
        analyze(config)
    else:
        raise ValueError(f"Unknown Phase 2 stage: {stage}")
