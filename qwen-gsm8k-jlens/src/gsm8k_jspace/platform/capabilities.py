"""PyTorch backend detection and capability probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BackendInfo:
    name: str
    device: str
    dtype: torch.dtype
    dtype_name: str
    torch_cuda_available: bool
    mps_available: bool
    hip_version: str | None
    cuda_version: str | None
    device_name: str | None = None
    warnings: list[str] = field(default_factory=list)


def detect_backend(requested: str = "auto") -> str:
    hip = getattr(torch.version, "hip", None)
    cuda_ver = getattr(torch.version, "cuda", None)
    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )

    detected = "cpu"
    if hip and cuda_available:
        detected = "rocm"
    elif cuda_ver and cuda_available:
        detected = "cuda"
    elif mps_available:
        detected = "mps"

    if requested != "auto" and requested != detected:
        if requested == "cpu":
            return "cpu"
        raise RuntimeError(
            f"requested backend {requested!r} but detected {detected!r} "
            f"(cuda_available={cuda_available}, mps_available={mps_available}, "
            f"hip={hip}, cuda={cuda_ver}). Refusing silent CPU fallback."
        )
    return detected


def resolve_dtype(requested: str, backend: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if requested != "auto":
        if requested not in mapping:
            raise ValueError(f"unknown dtype {requested!r}")
        return mapping[requested]
    if backend in {"cuda", "rocm"}:
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        return torch.bfloat16 if bf16_ok else torch.float16
    if backend == "mps":
        return torch.float16
    return torch.float32


def dtype_name(dtype: torch.dtype) -> str:
    for name, value in (
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
    ):
        if dtype == value:
            return name
    return str(dtype)


def probe_operations(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    results: dict[str, Any] = {}
    x = torch.randn(8, 8, device=device, dtype=dtype)
    y = torch.randn(8, 8, device=device, dtype=dtype)
    results["matmul"] = True
    _ = x @ y
    results["norm"] = True
    _ = torch.linalg.vector_norm(x.float())
    results["topk"] = True
    _ = x.float().reshape(-1).topk(3)
    cpu_eye = torch.eye(8, dtype=torch.float32)
    results["pinv_cpu"] = True
    _ = torch.linalg.pinv(cpu_eye)
    return results


def inspect_backend(requested: str = "auto", requested_dtype: str = "auto") -> BackendInfo:
    name = detect_backend(requested)
    resolved_dtype = resolve_dtype(requested_dtype, name)
    device = "cpu"
    device_name = None
    if name in {"cuda", "rocm"}:
        device = "cuda"
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
    elif name == "mps":
        device = "mps"
        device_name = "Apple MPS"
    warnings: list[str] = []
    if requested_dtype == "bfloat16" and name == "mps":
        warnings.append("bfloat16 is not a safe MPS default; using the requested dtype anyway")
    return BackendInfo(
        name=name,
        device=device,
        dtype=resolved_dtype,
        dtype_name=dtype_name(resolved_dtype),
        torch_cuda_available=bool(torch.cuda.is_available()),
        mps_available=bool(
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ),
        hip_version=getattr(torch.version, "hip", None),
        cuda_version=getattr(torch.version, "cuda", None),
        device_name=device_name,
        warnings=warnings,
    )
