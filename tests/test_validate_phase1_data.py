"""Tests for Phase 1 data requirement validation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "lib" / "validate_phase1_data.py"
spec = importlib.util.spec_from_file_location("validate_phase1_data", VALIDATOR_PATH)
assert spec and spec.loader
validate_phase1_data = importlib.util.module_from_spec(spec)
sys.modules["validate_phase1_data"] = validate_phase1_data
spec.loader.exec_module(validate_phase1_data)
validate_phase1_data_requirements = validate_phase1_data.validate_phase1_data_requirements


def _write_config(tmp_path: Path, tasks: list[str]) -> Path:
    config_path = tmp_path / "phase1.yaml"
    config_path.write_text(
        yaml.safe_dump({"tasks": tasks, "data": {"bipia_root": "BIPIA/benchmark"}}),
        encoding="utf-8",
    )
    return config_path


def test_smoke_email_task_passes_without_extra_files(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, ["email"])
    validate_phase1_data_requirements(
        config_path=config_path,
        repo_root=REPO_ROOT,
        bipia_root=REPO_ROOT / "BIPIA" / "benchmark",
    )


def test_full_qa_task_requires_webqa_train(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, ["qa"])
    with pytest.raises(FileNotFoundError, match="webqa_train_path"):
        validate_phase1_data_requirements(
            config_path=config_path,
            repo_root=REPO_ROOT,
            bipia_root=REPO_ROOT / "BIPIA" / "benchmark",
        )
