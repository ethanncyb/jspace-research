"""Backend detection, dtype/device resolution, and diagnostics."""

from __future__ import annotations

from gsm8k_jspace.platform.capabilities import (
    BackendInfo,
    detect_backend,
    probe_operations,
    resolve_dtype,
)
from gsm8k_jspace.platform.device import resolve_torch_device
from gsm8k_jspace.platform.diagnostics import collect_environment
from gsm8k_jspace.platform.memory import estimate_run_memory_gb, memory_preflight

__all__ = [
    "BackendInfo",
    "collect_environment",
    "detect_backend",
    "estimate_run_memory_gb",
    "memory_preflight",
    "probe_operations",
    "resolve_dtype",
    "resolve_torch_device",
]
