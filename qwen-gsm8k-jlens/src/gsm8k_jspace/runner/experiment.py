"""Unified experiment runner for baseline, no-op, and intervention."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gsm8k_jspace import SCHEMA_VERSION
from gsm8k_jspace.artifacts.manifest import (
    create_run_dir,
    finalize_manifest,
    load_completed_ids,
)
from gsm8k_jspace.artifacts.writer import append_jsonl, write_json
from gsm8k_jspace.capture.hooks import JSpaceCapture
from gsm8k_jspace.capture.selectors import resolve_layers
from gsm8k_jspace.config import AppConfig
from gsm8k_jspace.datasets.gsm8k import load_gsm8k_examples, selection_records
from gsm8k_jspace.interventions import build_controller
from gsm8k_jspace.models.jlens_adapter import load_jlens
from gsm8k_jspace.models.loader import load_model_bundle
from gsm8k_jspace.platform.diagnostics import collect_environment
from gsm8k_jspace.prompting.gsm8k import format_model_prompt, render_prompt
from gsm8k_jspace.runner.generation import generate_completion
from gsm8k_jspace.types import GSM8KExample


def _needs_lens(cfg: AppConfig) -> bool:
    return cfg.capture.enabled or cfg.experiment.condition in {"no_op", "intervention"}


def run_experiment(
    cfg: AppConfig,
    *,
    hf_model=None,
    tokenizer=None,
    examples: list[GSM8KExample] | None = None,
    project_root: Path | None = None,
) -> Path:
    if examples is None:
        examples = load_gsm8k_examples(cfg)
    selection = selection_records(examples)
    bundle = load_model_bundle(cfg, hf_model=hf_model, tokenizer=tokenizer)
    jlens = None
    lens_model = None
    capture_layers: list[int] = []
    intervention_layers: list[int] = []
    if _needs_lens(cfg):
        jlens, lens_model = load_jlens(cfg, bundle.hf_model, bundle.tokenizer)
        fitted = jlens.get_supported_layers()
        n_layers = getattr(lens_model, "n_layers", len(getattr(lens_model, "layers", [])))
        if cfg.capture.enabled:
            capture_layers = resolve_layers(cfg.capture.layers, n_layers, fitted)
        if cfg.experiment.condition in {"no_op", "intervention"}:
            intervention_layers = resolve_layers(
                cfg.intervention.layers, n_layers, fitted
            )

    environment = collect_environment(cfg, project_root=project_root)
    extra_manifest: dict[str, Any] = {
        "resolved_capture_layers": capture_layers,
        "resolved_intervention_layers": intervention_layers,
        "jlens": jlens.get_metadata() if jlens is not None else {"status": "disabled"},
    }
    run_dir = create_run_dir(
        cfg, selection, environment=environment, extra_manifest=extra_manifest
    )
    run_id = run_dir.name
    done = load_completed_ids(run_dir)
    if done:
        print(f"[run] resuming: {len(done)} examples already completed")

    controller = build_controller(
        cfg.experiment.condition,
        jlens=jlens,
        lens_model=lens_model,
        layers=intervention_layers,
        spec=cfg.intervention,
        compute_device=cfg.runtime.pinv.compute_device,
        log_path=(
            run_dir / "intervention" / "hook_log.jsonl"
            if cfg.experiment.condition == "intervention"
            else None
        ),
    )

    capture_index_path = run_dir / "captures" / "index.jsonl"
    n_written = len(done)
    try:
        with controller:
            for example in examples:
                if example.example_id in done:
                    continue
                prompt = format_model_prompt(
                    render_prompt(example, cfg), bundle.tokenizer, cfg
                )
                prompt_len = _prompt_len(bundle.tokenizer, prompt)
                controller.reset_example(example.example_id, prompt_len)
                capture = None
                if cfg.capture.enabled and jlens is not None and lens_model is not None:
                    capture = JSpaceCapture(
                        lens_model,
                        jlens,
                        layers=capture_layers,
                        capture_cfg=cfg.capture,
                        prompt_len=prompt_len,
                        run_id=run_id,
                        example_id=example.example_id,
                        condition=cfg.experiment.condition,
                    )
                live_capture = (
                    capture is not None
                    and cfg.capture.tokens.mode
                    not in JSpaceCapture.REPLAY_TOKEN_MODES
                )
                ctx = capture if live_capture else nullcontext()
                with ctx:
                    result = generate_completion(
                        prompt=prompt,
                        cfg=cfg,
                        bundle=bundle,
                        tokenizer=bundle.tokenizer,
                    )
                capture_rel = None
                if capture is not None:
                    full_ids = result.extra.get("full_ids")
                    if cfg.capture.tokens.mode == "full_sequence" and full_ids is not None:
                        capture.capture_sequence_replay(full_ids, bundle.tokenizer)
                    else:
                        capture.attach_tokens(result.generated_token_ids, bundle.tokenizer)
                        if (
                            cfg.capture.tokens.mode == "generated_last"
                            and cfg.prompt.final_token_capture == "replay"
                            and result.generated_token_ids
                        ):
                            capture.capture_final_replay(full_ids, bundle.tokenizer)
                    rel = f"captures/{example.example_id}.jsonl.gz"
                    index_row = capture.save(
                        run_dir / "captures" / f"{example.example_id}.jsonl"
                    )
                    index_row["path"] = rel
                    append_jsonl(capture_index_path, index_row)
                    capture_rel = rel
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "example_id": example.example_id,
                    "source_index": example.source_index,
                    "model": cfg.model.name,
                    "condition": cfg.experiment.condition,
                    "prompt_template": cfg.prompt.template,
                    "prompt": result.prompt if cfg.outputs.save_prompts else None,
                    "generated_text": result.generated_text,
                    "generated_token_ids": (
                        result.generated_token_ids
                        if cfg.outputs.save_generated_token_ids
                        else None
                    ),
                    "n_prompt_tokens": result.n_prompt_tokens,
                    "n_generated_tokens": len(result.generated_token_ids),
                    "finish_reason": result.finish_reason,
                    "seed": cfg.generation.seed,
                    "elapsed_seconds": result.elapsed_seconds,
                    "capture_file": capture_rel,
                    "gold_answer": example.gold_answer,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                append_jsonl(run_dir / "completions.jsonl", record)
                n_written += 1
                print(
                    f"[run] {n_written}/{len(examples)} {example.example_id} "
                    f"tokens={len(result.generated_token_ids)}"
                )
        write_json(run_dir / "intervention" / "summary.json", controller.summary())
        finalize_manifest(run_dir, completed_examples=n_written, status="complete")
    except Exception:
        finalize_manifest(run_dir, completed_examples=n_written, status="interrupted")
        raise
    print(f"[run] wrote {run_dir}")
    return run_dir


def _prompt_len(tokenizer, prompt: str) -> int:
    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return int(ids.shape[1])
