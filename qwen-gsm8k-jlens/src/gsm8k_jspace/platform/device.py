"""Device helpers that never assume NVIDIA CUDA."""

from __future__ import annotations

import torch

from gsm8k_jspace.platform.capabilities import inspect_backend


def resolve_torch_device(requested: str, backend: str) -> torch.device:
    if requested not in {"auto", ""}:
        return torch.device(requested)
    if backend in {"cuda", "rocm"}:
        return torch.device("cuda")
    if backend == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def tensor_device(module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def describe_runtime(requested_backend: str = "auto", requested_dtype: str = "auto"):
    info = inspect_backend(requested_backend, requested_dtype)
    device = resolve_torch_device("auto", info.name)
    return info, device
