"""Run directory and manifest lifecycle."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gsm8k_jspace import SCHEMA_VERSION
from gsm8k_jspace.artifacts.writer import read_json, read_jsonl, write_json, write_jsonl
from gsm8k_jspace.config import (
    AppConfig,
    condition_fingerprint,
    dump_resolved_yaml,
    run_fingerprint,
)


class ResumeError(ValueError):
    """Existing run is incompatible with the requested configuration."""


def make_run_id(cfg: AppConfig) -> str:
    if cfg.outputs.run_id:
        return cfg.outputs.run_id
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = run_fingerprint(cfg)[-8:]
    model_short = cfg.model.name.split("/")[-1].replace(".", "").lower()
    return f"{stamp}_{cfg.experiment.condition}_{model_short}_{digest}"


def dataset_selection_fingerprint(rows: list[dict[str, Any]]) -> str:
    payload = "".join(row["example_id"] for row in rows).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def create_run_dir(
    cfg: AppConfig,
    selection: list[dict[str, Any]],
    *,
    environment: dict[str, Any],
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    run_id = make_run_id(cfg)
    run_dir = Path(cfg.outputs.root_dir) / run_id
    manifest_path = run_dir / "manifest.json"
    if run_dir.exists() and manifest_path.exists():
        existing = read_json(manifest_path)
        expected = run_fingerprint(cfg)
        if existing.get("config_fingerprint") != expected:
            if cfg.outputs.on_existing != "resume":
                raise ResumeError(
                    f"run directory {run_dir} exists with a different fingerprint"
                )
            raise ResumeError(
                f"cannot resume {run_dir}: config fingerprint mismatch "
                f"({existing.get('config_fingerprint')} != {expected})"
            )
        if cfg.outputs.on_existing != "resume":
            raise ResumeError(
                f"run directory {run_dir} already exists; set outputs.on_existing=resume"
            )
        existing["status"] = "running"
        write_json(manifest_path, existing)
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "captures").mkdir(exist_ok=True)
    (run_dir / "intervention").mkdir(exist_ok=True)
    (run_dir / "evaluation").mkdir(exist_ok=True)
    write_jsonl(run_dir / "dataset_selection.jsonl", selection)
    (run_dir / "resolved_config.yaml").write_text(dump_resolved_yaml(cfg), encoding="utf-8")
    write_json(run_dir / "environment.json", environment)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "condition": cfg.experiment.condition,
        "test_type": cfg.benchmark.test_type,
        "config_fingerprint": run_fingerprint(cfg),
        "condition_fingerprint": condition_fingerprint(cfg),
        "dataset_selection_fingerprint": dataset_selection_fingerprint(selection),
        "model": {
            "name": cfg.model.name,
            "revision": cfg.model.revision,
            "dtype": cfg.model.dtype,
        },
        "jlens": extra_manifest.get("jlens") if extra_manifest else None,
        "resolved_capture_layers": extra_manifest.get("resolved_capture_layers", [])
        if extra_manifest
        else [],
        "resolved_intervention_layers": extra_manifest.get(
            "resolved_intervention_layers", []
        )
        if extra_manifest
        else [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "completed_examples": 0,
    }
    write_json(manifest_path, manifest)
    return run_dir


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "manifest.json")


def load_completed_ids(run_dir: Path) -> set[str]:
    return {row["example_id"] for row in read_jsonl(run_dir / "completions.jsonl")}


def finalize_manifest(run_dir: Path, *, completed_examples: int, status: str = "complete") -> None:
    manifest = load_manifest(run_dir)
    manifest["status"] = status
    manifest["completed_examples"] = completed_examples
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "manifest.json", manifest)
    write_json(
        run_dir / "progress.json",
        {"completed_examples": completed_examples, "status": status},
    )
