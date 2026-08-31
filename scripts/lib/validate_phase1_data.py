#!/usr/bin/env python3
"""Validate Phase 1 / Phase 4 data requirements for full BIPIA tasks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _is_file(path: str | Path | None) -> bool:
    return path is not None and Path(path).expanduser().is_file()


def validate_phase1_data_requirements(
    *,
    config_path: str | Path,
    repo_root: str | Path,
    bipia_root: str | Path | None = None,
    webqa_train_path: str | Path | None = None,
    summarization_train_path: str | Path | None = None,
) -> None:
    config_path = Path(config_path).expanduser().resolve()
    repo_root = Path(repo_root).resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks = tuple(raw.get("tasks", ()))
    data = raw.get("data", {})

    resolved_bipia_root = Path(bipia_root or data.get("bipia_root", "BIPIA/benchmark"))
    if not resolved_bipia_root.is_absolute():
        resolved_bipia_root = (repo_root / resolved_bipia_root).resolve()

    webqa = webqa_train_path or os.environ.get("WEBQA_TRAIN_PATH") or data.get("webqa_train_path")
    summary = (
        summarization_train_path
        or os.environ.get("SUMMARIZATION_TRAIN_PATH")
        or data.get("summarization_train_path")
    )

    errors: list[str] = []
    if "qa" in tasks:
        if not _is_file(webqa):
            errors.append(
                "Phase 1 task 'qa' (WebQA) needs a researcher-provided BIPIA-format train.jsonl. "
                "Run ./scripts/generate_bipia_jsonl.sh or set paths.webqa_train_path in "
                "scripts/config.yaml."
            )
        test_path = resolved_bipia_root / "qa" / "test.jsonl"
        if not test_path.is_file():
            errors.append(
                "Phase 4 also needs BIPIA/benchmark/qa/test.jsonl. "
                "Generate it with ./scripts/generate_bipia_jsonl.sh."
            )
    if "abstract" in tasks:
        if not _is_file(summary):
            errors.append(
                "Phase 1 task 'abstract' (Summarization) needs a researcher-provided "
                "BIPIA-format train.jsonl. Run ./scripts/generate_bipia_jsonl.sh or set "
                "paths.summarization_train_path in scripts/config.yaml."
            )
        test_path = resolved_bipia_root / "abstract" / "test.jsonl"
        if not test_path.is_file():
            errors.append(
                "Phase 4 also needs BIPIA/benchmark/abstract/test.jsonl. "
                "Generate it with ./scripts/generate_bipia_jsonl.sh."
            )

    if errors:
        lines = [f"Config: {config_path}", f"Tasks: {list(tasks)}", *errors]
        raise FileNotFoundError("\n".join(lines))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bipia-root", type=Path, default=None)
    parser.add_argument("--webqa-train", type=Path, default=None)
    parser.add_argument("--summarization-train", type=Path, default=None)
    args = parser.parse_args()
    validate_phase1_data_requirements(
        config_path=args.config,
        repo_root=args.repo_root,
        bipia_root=args.bipia_root,
        webqa_train_path=args.webqa_train,
        summarization_train_path=args.summarization_train,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
