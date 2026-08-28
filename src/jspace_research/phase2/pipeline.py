from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..model import HuggingFaceModelAdapter, load_tokenizer
from ..phase1.data import render_ids
from ..runtime import (
    append_jsonl,
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
ALPHA_TICK_LABELS = ("0\nintact", "0.5\npartial", "1.0\nfull removal")


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
    update_provenance(
        config.output_dir / "provenance.json",
        _base_provenance(config, handoff),
        defaults={
            "generation_gpu": None,
            "generation_packages": None,
            "analysis_packages": None,
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
        "schema_version": 1,
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
        "alpha": alpha,
        "alpha_index": alpha_index,
    }


# GPU stage: model loading, hooked generation, and resumable generation caches.
def generate(config: Phase2Config) -> Path:
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
            if expected["job_id"] in completed:
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
            append_jsonl(
                generation_path,
                {
                    **expected,
                    "generated_token_ids": token_ids,
                    "generation": generation,
                    "generation_sha256": _text_sha256(generation),
                    "zero_hook_equivalent": zero_hook_equivalent,
                },
            )
            completed.add(expected["job_id"])

    _load_generation_records(config, handoff)

    _write_or_validate_provenance(
        config,
        handoff,
        updates={
            "generation_gpu": cuda_metadata(
                model_input_device=str(model.input_device)
            ),
            "generation_packages": package_versions(PACKAGE_NAMES),
        },
    )
    del model, tokenizer, handoff
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
        if value.get("generation_sha256") != _text_sha256(value["generation"]):
            raise RuntimeError(f"Generation hash mismatch at {path}")
        if config.smoke and value["alpha"] == 0.0 and value.get(
            "zero_hook_equivalent"
        ) is not True:
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


def _expected_judgment_fields(
    config: Phase2Config, record: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "job_id": record["job_id"],
        "phase2_config_sha256": config.identity_hash(),
        "phase1_run_id": record["phase1_run_id"],
        "manifest_sha256": record["manifest_sha256"],
        "selected_layer": record["selected_layer"],
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
    label = judge.judge(record["attack_text"], record["generation"])
    if label not in {"YES", "NO", "UNKNOWN"}:
        raise RuntimeError(f"Judge returned an invalid label: {label!r}")
    value = {
        **expected,
        "judge_label": label,
        "attack_success": label == "YES",
    }
    append_jsonl(_judgment_path(config), value)
    cached_by_id[job_id] = value
    return value


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
    axis.set_xticklabels(ALPHA_TICK_LABELS)
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
    axis.set_xticklabels(ALPHA_TICK_LABELS)
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.output_dir / "phase2_clean_utility_vs_alpha.png", dpi=180)
    plt.close(figure)


# CPU/API stage: cached-generation scoring, judge calls, summaries, and plots.
def analyze(config: Phase2Config, *, judge: Any | None = None) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    handoff = load_phase1_handoff(config)
    _write_or_validate_provenance(config, handoff)
    records = _load_generation_records(config, handoff)
    judgments = _load_judgment_records(config, records)

    result_rows: list[dict[str, Any]] = []
    active_judge = judge
    for record in tqdm(records, desc="Phase 2 scoring"):
        attack_success: bool | None = None
        judge_label: str | None = None
        if record["condition"] == "attack":
            if active_judge is None and record["job_id"] not in judgments:
                active_judge = OpenAIAttackJudge(config.judge_model)
            judgment = _load_or_create_judgment(
                config=config,
                record=record,
                judge=active_judge,
                cached_by_id=judgments,
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
    atomic_write_parquet(results_path, results)

    summary = summarize_results(results)
    summary_path = config.output_dir / "phase2_summary.csv"
    atomic_write_csv(summary_path, summary)
    examples_path = config.output_dir / "phase2_examples.csv"
    atomic_write_csv(examples_path, qualitative_examples(results))
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
            "analysis_packages": existing_analysis_packages
            or package_versions(PACKAGE_NAMES),
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
