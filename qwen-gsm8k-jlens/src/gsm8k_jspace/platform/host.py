"""Host GPU detection that does not import the project virtualenv's torch."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class HostProfile:
    name: str
    backend: str
    venv_dir: str
    torch_backend: str | None
    uv_extra_args: tuple[str, ...] = ()


PROFILE_MAP = {
    "m1-max": HostProfile("m1-max", "mps", ".venv-mps", None),
    "radeon-8060s": HostProfile(
        "radeon-8060s", "rocm", ".venv-rocm", "auto", ("--torch-backend", "auto")
    ),
    "nvidia-datacenter": HostProfile(
        "nvidia-datacenter", "cuda", ".venv-cuda", "auto", ("--torch-backend", "auto")
    ),
    "cpu": HostProfile("cpu", "cpu", ".venv-cpu", "cpu", ("--torch-backend", "cpu")),
    "mps": HostProfile("m1-max", "mps", ".venv-mps", None),
    "rocm": HostProfile(
        "radeon-8060s", "rocm", ".venv-rocm", "auto", ("--torch-backend", "auto")
    ),
    "cuda": HostProfile(
        "nvidia-datacenter", "cuda", ".venv-cuda", "auto", ("--torch-backend", "auto")
    ),
}


def _run_ok(command: list[str]) -> bool:
    try:
        proc = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None and _run_ok(["nvidia-smi", "-L"])


def _has_amd() -> bool:
    if shutil.which("rocminfo") and _run_ok(["rocminfo"]):
        return True
    hip = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES")
    if hip is not None:
        return True
    return os.path.exists("/dev/kfd")


def detect_host_profile(explicit: str | None = None) -> HostProfile:
    if explicit and explicit != "auto":
        if explicit not in PROFILE_MAP:
            raise ValueError(f"unknown host profile {explicit!r}")
        return PROFILE_MAP[explicit]

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return PROFILE_MAP["m1-max"]
    if _has_nvidia():
        return PROFILE_MAP["nvidia-datacenter"]
    if _has_amd():
        return PROFILE_MAP["radeon-8060s"]
    return PROFILE_MAP["cpu"]
