"""Load saved run artifacts without touching the model."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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
        return warnings


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


def iter_capture_rows(run_dir: str | Path, *, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    cap_dir = Path(run_dir) / "captures"
    seen = 0
    for path in sorted(cap_dir.glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
                seen += 1
                if max_rows is not None and seen >= max_rows:
                    return
