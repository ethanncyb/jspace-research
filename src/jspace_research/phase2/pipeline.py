from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..model import HuggingFaceModelAdapter, load_tokenizer
from ..phase1.adapters import JacobianLensAdapter, validate_model_lens
from ..phase1.artifacts import resolve_selection
from ..phase1.data import render_ids
from ..phase1.jspace import build_normalized_dictionary
from ..runtime import (
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    cuda_metadata,
    ensure_cache_metadata,
    package_versions,
    read_json,
    read_resumable_jsonl,
    sha256_file,
    update_provenance,
    validate_identity_fields,
)
from .artifacts import Phase1Handoff, load_phase1_handoff
from .config import Phase2Config
from .plots import export_plot_data, save_run_overview
from .scoring import (
    JUDGE_GATEWAY,
    JUDGE_RUBRIC_SHA256,
    QUALITY_RUBRIC_SHA256,
    OpenRouterAttackJudge,
    OpenRouterQualityJudge,
    qualitative_examples,
    score_generation,
    summarize_quality,
    summarize_results,
    validate_quality,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


PACKAGE_NAMES = (
    "jspace-research",
    "jlens",
    "openai",
    "pandas",
    "rouge-score",
    "torch",
    "transformers",
)


def _run_id(config: Phase2Config, handoff: Phase1Handoff) -> str:
    return f"phase2-{config.identity_hash()[:12]}-{handoff.metadata['manifest_sha256'][:12]}"


def _base_provenance(config: Phase2Config, handoff: Phase1Handoff) -> dict[str, Any]:
    phase1_settings = config.phase1.scientific_dict()
    return {
        "schema_version": 2,
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
        "quality_judge": {"model": config.judge_model, "rubric_sha256": QUALITY_RUBRIC_SHA256},
        "judge": {
            "gateway": JUDGE_GATEWAY,
            "requested_model": config.judge_model,
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
    update_provenance(
        config.output_dir / "provenance.json",
        _base_provenance(config, handoff),
        defaults={
            "generation_gpu": None,
            "generation_packages": None,
            "analysis_packages": None,
            "judge_runtime": None,
        },
        updates=updates,
    )


def _generation_path(config: Phase2Config) -> Path:
    return config.output_dir / "generations.jsonl"


def _judgment_path(config: Phase2Config) -> Path:
    return config.output_dir / "judgments.jsonl"


def _job_id(example_index: int, alpha_index: int) -> str:
    return f"example_{example_index:06d}_alpha_{alpha_index}"


def _expected_generation_fields(
    config: Phase2Config,
    handoff: Phase1Handoff,
    example: dict[str, Any],
    alpha_index: int,
) -> dict[str, Any]:
    alpha = config.alphas[alpha_index]
    return {
        "schema_version": 2,
        "job_id": _job_id(int(example["example_index"]), alpha_index),
        "phase2_config_sha256": config.identity_hash(),
        "phase1_run_id": handoff.metadata["run_id"],
        "manifest_sha256": handoff.metadata["manifest_sha256"],
        "selected_layer": handoff.selected_layer,
        "example_id": f"{example['pair_id']}:{example['condition']}",
        "example_index": int(example["example_index"]),
        "pair_id": example["pair_id"],
        "task": example["task"],
        "condition": example["condition"],
        "prompt_hash": example["prompt_hash"],
        "K": config.phase1.sparsity_k,
        "W": config.output_window,
        "intervention": config.scientific_dict()["intervention"],
        "alpha": alpha,
        "alpha_index": alpha_index,
    }


# GPU stage: model loading, hooked generation, and resumable generation caches.
def generate(config: Phase2Config, *, resources: dict[str, Any] | None = None) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("The Phase 2 generate stage requires a CUDA GPU")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    handoff = load_phase1_handoff(config)
    _write_or_validate_provenance(config, handoff)
    generation_path = _generation_path(config)
    cached_records = _load_generation_records(config, handoff, require_complete=False)
    completed = {record["job_id"] for record in cached_records}
    expected_count = len(handoff.validation_examples) * len(config.alphas)
    if len(completed) == expected_count:
        provenance = read_json(config.output_dir / "provenance.json")
        updates: dict[str, Any] = {}
        if provenance.get("generation_gpu") is None:
            updates["generation_gpu"] = cuda_metadata(model_input_device=None)
        if provenance.get("generation_packages") is None:
            updates["generation_packages"] = package_versions(PACKAGE_NAMES)
        if updates:
            _write_or_validate_provenance(config, handoff, updates=updates)
        print(f"Phase 2 generation cache already complete: {generation_path}")
        return generation_path

    shared = resources if resources is not None else {}
    if "model" not in shared:
        shared["tokenizer"] = load_tokenizer(config.phase1)
        shared["model"] = HuggingFaceModelAdapter.load(config.phase1, shared["tokenizer"])
    tokenizer, model = shared["tokenizer"], shared["model"]
    if model.hidden_width != handoff.reconstruction_shape[1]:
        raise RuntimeError("Model width does not match the selected-layer reconstruction cache")
    dictionary_key = (config.phase1.lens.sha256, handoff.selected_layer)
    if shared.get("dictionary_key") != dictionary_key:
        shared.pop("dictionary", None)
        lens = JacobianLensAdapter.load(config.phase1)
        validate_model_lens(model, lens)
        shared["dictionary"] = build_normalized_dictionary(
            jacobian=lens.jacobian(handoff.selected_layer),
            unembedding=model.unembedding(),
            layer=handoff.selected_layer,
            device=model.input_device,
            chunk_size=config.phase1.dictionary_chunk_size,
        )
        shared["dictionary_key"] = dictionary_key
    dictionary = shared["dictionary"]

    for example in tqdm(handoff.validation_examples, desc="Phase 2 intervention generation"):
        example_index = int(example["example_index"])
        input_ids = render_ids(tokenizer, example["messages"])
        if int(input_ids.shape[-1]) > config.phase1.max_input_tokens:
            raise RuntimeError(f"Frozen example {example_index} exceeds max_input_tokens")

        for alpha_index, alpha in enumerate(config.alphas):
            expected = _expected_generation_fields(config, handoff, example, alpha_index)
            if expected["job_id"] in completed:
                continue

            intervention_stats: dict[str, int] = {}
            generated_ids = model.generate_from_prompt(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                layer=handoff.selected_layer,
                dictionary=dictionary,
                sparsity_k=config.phase1.sparsity_k,
                screen_candidates=config.phase1.screen_candidates,
                output_window=config.output_window,
                intervention_stats=intervention_stats,
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
            append_jsonl(
                generation_path,
                {
                    **expected,
                    "generated_token_ids": token_ids,
                    "generation": generation,
                    "generation_sha256": _text_sha256(generation),
                    "zero_hook_equivalent": zero_hook_equivalent,
                    **intervention_stats,
                },
            )
            completed.add(expected["job_id"])

    _load_generation_records(config, handoff)

    _write_or_validate_provenance(
        config,
        handoff,
        updates={
            "generation_gpu": cuda_metadata(model_input_device=str(model.input_device)),
            "generation_packages": package_versions(PACKAGE_NAMES),
        },
    )
    del model, tokenizer, handoff, dictionary
    if resources is None:
        shared.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Phase 2 generation cache complete: {generation_path}")
    return generation_path


def _load_generation_records(
    config: Phase2Config,
    handoff: Phase1Handoff,
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    path = _generation_path(config)
    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for example in handoff.validation_examples:
        for alpha_index in range(len(config.alphas)):
            expected = _expected_generation_fields(config, handoff, example, alpha_index)
            jobs.append((example, expected))

    expected_by_id = {expected["job_id"]: expected for _, expected in jobs}
    cached_by_id: dict[str, dict[str, Any]] = {}
    for value in read_resumable_jsonl(path):
        job_id = value.get("job_id")
        if not isinstance(job_id, str) or job_id not in expected_by_id:
            raise RuntimeError(f"Unexpected generation cache job at {path}")
        if job_id in cached_by_id:
            raise RuntimeError(f"Duplicate generation cache job {job_id} at {path}")
        validate_identity_fields(path, value, expected_by_id[job_id])
        if not isinstance(value.get("generated_token_ids"), list) or not isinstance(
            value.get("generation"), str
        ):
            raise RuntimeError(f"Generation cache is incomplete at {path}")
        processed, edited = value.get("processed_output_tokens"), value.get("edited_output_tokens")
        expected_processed = min(
            config.output_window, max(0, len(value["generated_token_ids"]) - 1)
        )
        if (
            type(processed) is not int
            or processed != expected_processed
            or type(edited) is not int
            or edited != (0 if value["alpha"] == 0 else processed)
        ):
            raise RuntimeError(f"Generation intervention counts are invalid at {path}")
        if value.get("generation_sha256") != _text_sha256(value["generation"]):
            raise RuntimeError(f"Generation hash mismatch at {path}")
        if config.smoke and value["alpha"] == 0.0 and value.get("zero_hook_equivalent") is not True:
            raise RuntimeError(f"Smoke zero-hook equivalence is missing at {path}")
        cached_by_id[job_id] = value

    if require_complete:
        missing = [job_id for job_id in expected_by_id if job_id not in cached_by_id]
        if missing:
            raise RuntimeError(
                f"Phase 2 generation is incomplete; {len(missing)} jobs are missing from {path}"
            )
    return [
        {**example, **cached_by_id[expected["job_id"]]}
        for example, expected in jobs
        if expected["job_id"] in cached_by_id
    ]


def _expected_judgment_fields(config: Phase2Config, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "job_id": record["job_id"],
        "phase2_config_sha256": config.identity_hash(),
        "phase1_run_id": record["phase1_run_id"],
        "manifest_sha256": record["manifest_sha256"],
        "selected_layer": record["selected_layer"],
        "judge_gateway": JUDGE_GATEWAY,
        "judge_model": config.judge_model,
        "judge_rubric_sha256": JUDGE_RUBRIC_SHA256,
        "generation_sha256": record["generation_sha256"],
        "attack_text_sha256": _text_sha256(record["attack_text"]),
    }


def _load_judgment_records(
    config: Phase2Config, generation_records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    path = _judgment_path(config)
    expected_by_id = {
        record["job_id"]: _expected_judgment_fields(config, record)
        for record in generation_records
        if record["condition"] == "attack"
    }
    cached_by_id: dict[str, dict[str, Any]] = {}
    for value in read_resumable_jsonl(path):
        job_id = value.get("job_id")
        if not isinstance(job_id, str) or job_id not in expected_by_id:
            raise RuntimeError(f"Unexpected judgment cache job at {path}")
        if job_id in cached_by_id:
            raise RuntimeError(f"Duplicate judgment cache job {job_id} at {path}")
        validate_identity_fields(path, value, expected_by_id[job_id])
        if value.get("judge_label") not in {"YES", "NO", "UNKNOWN"}:
            raise RuntimeError(f"Judgment cache is incomplete at {path}")
        if value.get("attack_success") != (value["judge_label"] == "YES"):
            raise RuntimeError(f"Judgment cache outcome is inconsistent at {path}")
        for field in ("returned_model", "provider", "provider_model"):
            if value.get(field) is not None and not isinstance(value[field], str):
                raise RuntimeError(f"Judgment cache metadata is invalid at {path}")
        cached_by_id[job_id] = value
    return cached_by_id


def _load_or_create_judgment(
    *,
    config: Phase2Config,
    record: dict[str, Any],
    judge: Any,
    cached_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    job_id = record["job_id"]
    if job_id in cached_by_id:
        return cached_by_id[job_id]
    expected = _expected_judgment_fields(config, record)
    outcome = judge.judge(record["attack_text"], record["generation"])
    if outcome.label not in {"YES", "NO", "UNKNOWN"}:
        raise RuntimeError(f"Judge returned an invalid label: {outcome.label!r}")
    value = {
        **expected,
        "judge_label": outcome.label,
        "attack_success": outcome.label == "YES",
        "returned_model": outcome.returned_model,
        "provider": outcome.provider,
        "provider_model": outcome.provider_model,
    }
    append_jsonl(_judgment_path(config), value)
    cached_by_id[job_id] = value
    return value


def _judge_runtime_metadata(
    judgments: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    return {
        provenance_field: sorted(
            {
                str(value[cache_field])
                for value in judgments.values()
                if value.get(cache_field) is not None
            }
        )
        for provenance_field, cache_field in (
            ("returned_models", "returned_model"),
            ("providers", "provider"),
            ("provider_models", "provider_model"),
        )
    }


def _save_plots(config: Phase2Config, summary: pd.DataFrame) -> None:
    asr = summary[
        (summary.metric == "asr") & (summary.scope == "overall") & (summary.condition == "attack")
    ].sort_values("alpha")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(asr.alpha, asr.value, marker="o")
    axis.set_xlabel("J-space removal strength (alpha)")
    axis.set_ylabel("Attack Success Rate")
    axis.set_title("Phase 2 Attack Success vs J-Space Removal")
    axis.set_xticks(list(config.alphas))
    axis.set_xticklabels([str(alpha) for alpha in config.alphas])
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
    axis.set_xticklabels([str(alpha) for alpha in config.alphas])
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.output_dir / "phase2_clean_utility_vs_alpha.png", dpi=180)
    plt.close(figure)


# CPU/API stage: cached-generation scoring, judge calls, summaries, and plots.
def analyze(
    config: Phase2Config, *, judge: Any | None = None, quality_judge: Any | None = None
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    handoff = load_phase1_handoff(config)
    _write_or_validate_provenance(config, handoff)
    records = _load_generation_records(config, handoff)
    judgments = _load_judgment_records(config, records)
    quality_judgments = _load_quality_records(config, records)

    result_rows: list[dict[str, Any]] = []
    active_judge = judge
    for record in tqdm(records, desc="Phase 2 scoring"):
        attack_success: bool | None = None
        judge_label: str | None = None
        if record["condition"] == "attack":
            if active_judge is None and record["job_id"] not in judgments:
                active_judge = OpenRouterAttackJudge(config.judge_model)
            judgment = _load_or_create_judgment(
                config=config,
                record=record,
                judge=active_judge,
                cached_by_id=judgments,
            )
            attack_success = bool(judgment["attack_success"])
            judge_label = str(judgment["judge_label"])
        if quality_judge is None and record["job_id"] not in quality_judgments:
            quality_judge = OpenRouterQualityJudge(config.judge_model)
        quality = _quality_judgment(config, record, quality_judge, quality_judgments)
        scores = score_generation(record["generation"], record["target"])
        result_rows.append(
            {
                "K": config.phase1.sparsity_k,
                "W": config.output_window,
                "selected_layer": handoff.selected_layer,
                "processed_output_tokens": record["processed_output_tokens"],
                "edited_output_tokens": record["edited_output_tokens"],
                "garbage_label": quality["garbage_label"],
                "degradation_severity": quality["degradation_severity"],
                "quality_explanation": quality["explanation"],
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
        "K",
        "W",
        "selected_layer",
        "processed_output_tokens",
        "edited_output_tokens",
        "garbage_label",
        "degradation_severity",
        "quality_explanation",
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
    atomic_write_parquet(results_path, results)

    summary = pd.concat([summarize_results(results), summarize_quality(results)], ignore_index=True)
    summary = summary.assign(
        K=config.phase1.sparsity_k, W=config.output_window, selected_layer=handoff.selected_layer
    )
    summary_path = config.output_dir / "phase2_summary.csv"
    atomic_write_csv(summary_path, summary)
    examples_path = config.output_dir / "phase2_examples.csv"
    atomic_write_csv(
        examples_path,
        qualitative_examples(results).assign(K=config.phase1.sparsity_k, W=config.output_window),
    )
    _save_plots(config, summary)
    _save_quality_plots(config.output_dir, summary)
    export_path = export_plot_data(
        config.output_dir,
        results,
        summary,
        [_base_provenance(config, handoff)],
        label=f"K={config.phase1.sparsity_k}, W={config.output_window}",
    )
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
                "plot_data": {"path": export_path.name, "sha256": sha256_file(export_path)},
                "jsonl_results": {
                    "path": "phase2_results.jsonl",
                    "sha256": sha256_file(config.output_dir / "phase2_results.jsonl"),
                },
                "asr_plot": "phase2_asr_vs_alpha.png",
                "utility_plot": "phase2_clean_utility_vs_alpha.png",
                "quality_plot": "phase2_output_quality_vs_alpha.png",
            },
            "analysis_packages": existing_analysis_packages or package_versions(PACKAGE_NAMES),
            "judge_runtime": _judge_runtime_metadata(judgments),
            "quality_judge_runtime": _judge_runtime_metadata(quality_judgments),
        },
    )
    print(f"Phase 2 results: {results_path}")
    return results_path


def _quality_identity(config: Phase2Config, record: dict[str, Any]) -> dict[str, Any]:
    return {
        **_expected_judgment_fields(config, record),
        "judge_rubric_sha256": QUALITY_RUBRIC_SHA256,
        "messages_sha256": _text_sha256(
            json.dumps(record["messages"], sort_keys=True, ensure_ascii=False)
        ),
        "K": config.phase1.sparsity_k,
        "W": config.output_window,
    }


def _load_quality_records(
    config: Phase2Config, records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    path = config.output_dir / "quality_judgments.jsonl"
    expected = {r["job_id"]: _quality_identity(config, r) for r in records}
    cached = {}
    for value in read_resumable_jsonl(path):
        job_id = value.get("job_id")
        if job_id not in expected or job_id in cached:
            raise RuntimeError(f"Unexpected or duplicate quality judgment at {path}")
        validate_identity_fields(path, value, expected[job_id])
        validate_quality(value)
        for field in ("returned_model", "provider", "provider_model"):
            if value.get(field) is not None and not isinstance(value[field], str):
                raise RuntimeError(f"Invalid quality judge routing metadata at {path}")
        cached[job_id] = value
    return cached


def _quality_judgment(
    config: Phase2Config, record: dict[str, Any], judge: Any, cached: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    job_id = record["job_id"]
    if job_id not in cached:
        outcome = asdict(judge.judge(record["messages"], record["generation"]))
        validate_quality(outcome)
        value = {**_quality_identity(config, record), **outcome}
        append_jsonl(config.output_dir / "quality_judgments.jsonl", value)
        cached[job_id] = value
    return cached[job_id]


def _save_quality_plots(output_dir: Path, summary: pd.DataFrame) -> None:
    # Separate task panels keep task-specific degradation visible.
    for scope in [None, *sorted(summary.task.dropna().unique())]:
        frame = (
            summary[summary.scope == "overall"]
            if scope is None
            else summary[(summary.scope == "task") & (summary.task == scope)]
        )
        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        for axis, metric in zip(axes, ("garbage_rate", "mean_degradation_severity"), strict=True):
            for (k, w, condition), group in frame[frame.metric == metric].groupby(
                ["K", "W", "condition"]
            ):
                group = group.sort_values("alpha")
                axis.plot(group.alpha, group.value, marker="o", label=f"K={k}, W={w}, {condition}")
            axis.set_xlabel("J-space removal strength (alpha)")
            axis.set_ylabel(metric.replace("_", " "))
            axis.set_title("Overall" if scope is None else str(scope))
        axes[-1].legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
        figure.tight_layout()
        suffix = "" if scope is None else f"_{scope}"
        figure.savefig(
            output_dir / f"phase2_output_quality_vs_alpha{suffix}.png", dpi=180, bbox_inches="tight"
        )
        plt.close(figure)


def sweep_configs(config: Phase2Config) -> list[Phase2Config]:
    metadata = read_json(config.phase1_selected_path)
    ks = config.phase1.k_values or (config.phase1.sparsity_k,)
    if metadata.get("kind") != "k_sweep":
        _, k = resolve_selection(config.phase1_selected_path)
        if k not in ks:
            raise ValueError(f"Handoff K={k} is not in the configured K list")
        ks = (k,)
    children = []
    for k in ks:
        selected_path, _ = resolve_selection(config.phase1_selected_path, k)
        for w in config.windows or (config.output_window,):
            children.append(
                replace(
                    config,
                    phase1=replace(config.phase1, sparsity_k=k),
                    phase1_selected_path=selected_path,
                    output_window=w,
                    windows=(),
                    output_dir=config.output_dir / f"k{k}" / f"w{w}",
                )
            )
    return children


def _combine_results(config: Phase2Config, children: list[Phase2Config]) -> None:
    results = pd.concat(
        [pd.read_parquet(c.output_dir / "phase2_results.parquet") for c in children],
        ignore_index=True,
    )
    summary = pd.concat(
        [pd.read_csv(c.output_dir / "phase2_summary.csv") for c in children], ignore_index=True
    )
    examples = pd.concat(
        [pd.read_csv(c.output_dir / "phase2_examples.csv") for c in children], ignore_index=True
    )
    atomic_write_parquet(config.output_dir / "phase2_results.parquet", results)
    atomic_write_csv(config.output_dir / "phase2_summary.csv", summary)
    atomic_write_csv(config.output_dir / "phase2_examples.csv", examples)
    _save_quality_plots(config.output_dir, summary)
    export_path = export_plot_data(
        config.output_dir,
        results,
        summary,
        [read_json(c.output_dir / "provenance.json") for c in children],
        label=config.output_dir.parent.name,
    )
    save_run_overview(read_json(export_path), config.output_dir)
    for metric, filename, condition in (
        ("asr", "phase2_asr_vs_alpha.png", "attack"),
        ("rougeL_recall", "phase2_clean_utility_vs_alpha.png", "control"),
    ):
        frame = summary[(summary.metric == metric) & (summary.condition == condition)]
        frame = frame[frame.scope == ("overall" if metric == "asr" else "task")]
        figure, axis = plt.subplots(figsize=(10, 6))
        for (k, w, task), group in frame.groupby(["K", "W", "task"], dropna=False):
            group = group.sort_values("alpha")
            axis.plot(
                group.alpha,
                group.value,
                marker="o",
                label=f"K={k}, W={w}" + (f", {task}" if pd.notna(task) else ""),
            )
        axis.set_xlabel("J-space removal strength (alpha)")
        axis.set_ylabel(metric)
        axis.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
        figure.tight_layout()
        figure.savefig(config.output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)


def run(config: Phase2Config, stage: str) -> None:
    if stage not in {"generate", "analyze", "all"}:
        raise ValueError(f"Unknown Phase 2 stage: {stage}")
    config.validate()
    if (config.output_dir / "generations.jsonl").exists():
        raise RuntimeError(
            "Legacy flat Phase 2 output directory; use a new directory for the K/W sweep"
        )
    children = sweep_configs(config)
    ensure_cache_metadata(
        config.output_dir / "sweep_config.json",
        {
            "schema_version": 2,
            "combinations": [
                {"K": c.phase1.sparsity_k, "W": c.output_window, "config_sha256": c.identity_hash()}
                for c in children
            ],
        },
    )
    resources: dict[str, Any] = {}
    try:
        if stage in {"generate", "all"}:
            for child in children:
                generate(child, resources=resources)
    finally:
        resources.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if stage in {"analyze", "all"}:
        for child in children:
            analyze(child)
        _combine_results(config, children)
    atomic_write_json(
        config.output_dir / "sweep_index.json",
        {
            "schema_version": 2,
            "phase": 2,
            "combinations": [
                {
                    "K": c.phase1.sparsity_k,
                    "W": c.output_window,
                    "path": str(c.output_dir.relative_to(config.output_dir)),
                    "config_sha256": c.identity_hash(),
                }
                for c in children
            ],
        },
    )
