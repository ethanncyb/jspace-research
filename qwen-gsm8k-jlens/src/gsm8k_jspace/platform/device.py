"""Device helpers that never assume NVIDIA CUDA."""

from __future__ import annotations

import os

import torch

from gsm8k_jspace.config import AppConfig
from gsm8k_jspace.platform.capabilities import inspect_backend


def resolve_torch_device(requested: str, backend: str) -> torch.device:
    if requested in {"mlx"} or backend == "mlx":
        return torch.device("cpu")
    if requested not in {"auto", ""}:
        return torch.device(requested)
    if backend in {"cuda", "rocm"}:
        return torch.device("cuda")
    if backend == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def resolve_gpus(cfg: AppConfig) -> list[int]:
    """Return physical GPU indices to use; empty config means ``[0]``."""
    return list(cfg.runtime.gpus) if cfg.runtime.gpus else [0]


def should_run_parallel(cfg: AppConfig, gpus: list[int] | None = None) -> bool:
    ids = list(gpus) if gpus is not None else resolve_gpus(cfg)
    return bool(cfg.runtime.parallel and len(ids) > 1)


def pin_visible_gpu(gpu_id: int) -> None:
    """Restrict this process to one physical GPU before CUDA initializes."""
    value = str(int(gpu_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    os.environ["HIP_VISIBLE_DEVICES"] = value


def prepare_worker_device_config(cfg: AppConfig) -> AppConfig:
    """Force whole-model placement on the single visible CUDA device."""
    cfg.runtime.device = "cuda"
    if cfg.model.device_map in {"auto", "", None}:
        cfg.model.device_map = "cuda"
    return cfg


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
