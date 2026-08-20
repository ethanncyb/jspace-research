"""Load saved run artifacts without touching the model."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gsm8k_jspace.artifacts.writer import read_json, read_jsonl
from gsm8k_jspace.config import AppConfig, load_config


@dataclass
class RunArtifacts:
    run_dir: Path
    manifest: dict[str, Any]
    environment: dict[str, Any]
    config: AppConfig
    completions: list[dict[str, Any]]
    evaluation: list[dict[str, Any]]
    summary: dict[str, Any] | None
    selection: list[dict[str, Any]]

    def warn_incomplete(self) -> list[str]:
        warnings = []
        if self.manifest.get("status") != "complete":
            warnings.append(f"run status is {self.manifest.get('status')!r}")
        if self.environment.get("backend", {}).get("resolved") == "cpu":
            if self.config.runtime.backend not in {"cpu", "auto"}:
                warnings.append("run resolved to CPU unexpectedly")
        if self.environment.get("jlens", {}).get("placeholder"):
            warnings.append("identity J-Lens placeholder was used")
        if self.config.capture.enabled and not self.has_captures():
            warnings.append("capture was enabled but no capture files were found")
        return warnings

    def has_captures(self) -> bool:
        cap_dir = self.run_dir / "captures"
        return cap_dir.is_dir() and any(cap_dir.glob("*.jsonl.gz"))


def load_run(run_dir: str | Path) -> RunArtifacts:
    run_dir = Path(run_dir)
    summary_path = run_dir / "evaluation" / "summary.json"
    env = {}
    if (run_dir / "environment.json").exists():
        env = read_json(run_dir / "environment.json")
    return RunArtifacts(
        run_dir=run_dir,
        manifest=read_json(run_dir / "manifest.json"),
        environment=env,
        config=load_config(run_dir / "resolved_config.yaml"),
        completions=read_jsonl(run_dir / "completions.jsonl"),
        evaluation=read_jsonl(run_dir / "evaluation" / "results.jsonl"),
        summary=read_json(summary_path) if summary_path.exists() else None,
        selection=read_jsonl(run_dir / "dataset_selection.jsonl"),
    )


def _capture_paths(cap_dir: Path, example_id: str | None) -> list[Path]:
    if example_id is not None:
        for name in (f"{example_id}.jsonl.gz", f"{example_id}.jsonl"):
            path = cap_dir / name
            if path.exists():
                return [path]
        return []
    return sorted(
        path
        for path in cap_dir.glob("*.jsonl.gz")
        if path.name != "index.jsonl.gz"
    )


def _open_capture(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def iter_capture_rows(
    run_dir: str | Path,
    *,
    example_id: str | None = None,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    cap_dir = Path(run_dir) / "captures"
    if not cap_dir.is_dir():
        return
    seen = 0
    for path in _capture_paths(cap_dir, example_id):
        with _open_capture(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if example_id is not None and row.get("example_id") not in {None, example_id}:
                    continue
                yield row
                seen += 1
                if max_rows is not None and seen >= max_rows:
                    return
