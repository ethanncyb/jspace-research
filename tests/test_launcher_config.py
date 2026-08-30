"""Tests for the unified launcher config loader."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOADER = REPO_ROOT / "scripts" / "lib" / "load_launcher_config.py"


def _run_loader(*args: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("JSPACE_") or key in {"WEBQA_TRAIN_PATH", "SUMMARIZATION_TRAIN_PATH", "SKIP_SETUP", "SKIP_GPU_CHECK", "STAGE"}:
            del env[key]
    result = subprocess.run(
        [sys.executable, str(LOADER), "--repo-root", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    exports: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.startswith("export "):
            continue
        name, _, value = line[len("export ") :].partition("=")
        exports[name] = value.strip("'")
    return exports


def test_unified_config_qwen_smoke_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  config: configs/phase1_qwen35_9b_smoke.yaml
hardware:
  physical_gpu_index: 0
output:
  run_root: null
paths:
  benchmarks_root: ../jspace-benchmarks
  bipia_checkout: BIPIA
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exports = _run_loader("--config", str(config_path))

    assert exports["JSPACE_CONFIG_PATH"].endswith("configs/phase1_qwen35_9b_smoke.yaml")
    assert exports["JSPACE_PHYSICAL_GPU_INDEX"] == "0"
    assert exports["JSPACE_RUN_ROOT"].endswith("artifacts/jspace-qwen35_9b-smoke-gpu0")
    assert exports["JSPACE_PHASE1_DIR"].endswith("artifacts/jspace-qwen35_9b-smoke-gpu0/phase1")
    assert exports["JSPACE_PHASE2_DIR"].endswith("/phase2")
    assert exports["JSPACE_BIPIA_ROOT"].endswith("BIPIA/benchmark")


def test_explicit_run_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  config: configs/phase1_qwen35_9b_smoke.yaml
hardware:
  physical_gpu_index: 2
output:
  run_root: artifacts/my-custom-run
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exports = _run_loader("--config", str(config_path))

    assert exports["JSPACE_PHYSICAL_GPU_INDEX"] == "2"
    assert exports["JSPACE_RUN_ROOT"].endswith("artifacts/my-custom-run")
    assert exports["JSPACE_PHASE3_DIR"].endswith("artifacts/my-custom-run/phase3")


def test_local_config_merge(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    local_path = tmp_path / "config.local.yaml"
    config_path.write_text(
        """
experiment:
  config: configs/phase1_qwen35_9b_smoke.yaml
hardware:
  physical_gpu_index: 0
output:
  run_root: null
""".strip()
        + "\n",
        encoding="utf-8",
    )
    local_path.write_text(
        """
hardware:
  physical_gpu_index: 3
output:
  run_root: artifacts/overridden-run
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exports = _run_loader("--config", str(config_path), "--local-config", str(local_path))

    assert exports["JSPACE_PHYSICAL_GPU_INDEX"] == "3"
    assert exports["JSPACE_RUN_ROOT"].endswith("artifacts/overridden-run")


def test_derive_run_name_gemma_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  config: configs/phase1_smoke.yaml
hardware:
  physical_gpu_index: 1
output:
  run_root: null
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exports = _run_loader("--config", str(config_path))

    assert exports["JSPACE_RUN_ROOT"].endswith("artifacts/jspace-smoke-gpu1")


def test_committed_config_loads() -> None:
    config_path = REPO_ROOT / "scripts" / "config.yaml"
    if not config_path.is_file():
        pytest.skip("committed scripts/config.yaml not present")

    exports = _run_loader("--config", str(config_path))

    assert "phase1_qwen35_9b_smoke.yaml" in exports["JSPACE_CONFIG_PATH"]
    assert exports["JSPACE_PHYSICAL_GPU_INDEX"] == "0"
