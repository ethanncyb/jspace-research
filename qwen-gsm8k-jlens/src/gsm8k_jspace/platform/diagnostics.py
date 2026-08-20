"""Normalized environment metadata for run manifests."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gsm8k_jspace import SCHEMA_VERSION, __version__
from gsm8k_jspace.config import AppConfig, condition_fingerprint, run_fingerprint
from gsm8k_jspace.platform.capabilities import inspect_backend
from gsm8k_jspace.platform.host import detect_host_profile
from gsm8k_jspace.platform.memory import estimate_run_memory_gb


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _uv_version() -> str | None:
    try:
        proc = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _lock_sha(root: Path | None) -> str | None:
    if root is None:
        return None
    lock = root / "uv.lock"
    if not lock.exists():
        return None
    import hashlib

    return "sha256:" + hashlib.sha256(lock.read_bytes()).hexdigest()


def collect_environment(
    cfg: AppConfig,
    *,
    project_root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    host = detect_host_profile(cfg.runtime.host_profile)
    info = inspect_backend(cfg.runtime.backend, cfg.model.dtype)
    estimate = estimate_run_memory_gb(cfg)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "gsm8k_jspace": __version__,
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "datasets": _package_version("datasets"),
            "jlens": _package_version("jlens"),
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm"),
        },
        "backend": {
            "requested": cfg.runtime.backend,
            "resolved": info.name,
            "device": info.device,
            "dtype_requested": cfg.model.dtype,
            "dtype_resolved": info.dtype_name,
            "device_name": info.device_name,
            "hip_version": info.hip_version,
            "cuda_version": info.cuda_version,
            "mps_available": info.mps_available,
            "mlx_available": info.mlx_available,
            "warnings": info.warnings,
        },
        "host_profile": host.name,
        "run_tier": cfg.experiment.run_tier,
        "uv_version": _uv_version(),
        "uv_project_environment": os.environ.get(
            "UV_PROJECT_ENVIRONMENT", host.venv_dir
        ),
        "uv_torch_backend_requested": host.torch_backend,
        "uv_lock_sha256": _lock_sha(project_root),
        "run_fingerprint": run_fingerprint(cfg),
        "condition_fingerprint": condition_fingerprint(cfg),
        "memory_estimate_gb": estimate.__dict__,
        "pinv": {
            "compute_device": cfg.runtime.pinv.compute_device,
            "cache_device": cfg.runtime.pinv.cache_device,
            "preload": cfg.runtime.pinv.preload,
        },
        "compatibility_mode": cfg.runtime.compatibility_mode,
    }
    if extra:
        payload.update(extra)
    return payload
