"""Coarse memory estimates and preflight checks."""

from __future__ import annotations

from dataclasses import dataclass

from gsm8k_jspace.config import AppConfig


DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "auto": 2}


@dataclass
class MemoryEstimate:
    model_gb: float
    lens_gb: float
    pinv_gb: float
    kv_cache_gb: float
    capture_gb: float
    total_gb: float
    notes: list[str]


def _bytes_to_gb(n_bytes: float) -> float:
    return n_bytes / (1024 ** 3)


def estimate_run_memory_gb(
    cfg: AppConfig,
    *,
    n_layers: int = 32,
    d_model: int = 4096,
    n_fitted_layers: int | None = None,
) -> MemoryEstimate:
    notes: list[str] = []
    width = d_model
    layers = n_layers
    fitted = n_fitted_layers if n_fitted_layers is not None else layers
    itemsize = DTYPE_BYTES.get(cfg.model.dtype, 2)

    # Rough 9B-class default when the configured name looks like 9B.
    n_params = 9.0e9 if "9B" in cfg.model.name.upper() else width * width * layers * 12
    model_bytes = n_params * itemsize
    lens_bytes = fitted * width * width * 4  # float32 jacobians
    pinv_layers = fitted if cfg.runtime.pinv.preload else min(fitted, 8)
    pinv_bytes = 0.0
    if cfg.experiment.condition == "intervention" or cfg.intervention.enabled:
        pinv_bytes = pinv_layers * width * width * 4
        if cfg.runtime.pinv.compute_device == "cpu":
            notes.append("pseudo-inverses computed/cached on CPU by default")
    kv_bytes = cfg.generation.max_new_tokens * layers * 2 * width * itemsize
    capture_bytes = 0.0
    if cfg.capture.enabled:
        rows = max(cfg.benchmark.subset_size, 1) * max(cfg.generation.max_new_tokens, 1) * 8
        capture_bytes = rows * 64
        if cfg.capture.fields.hidden_vector or cfg.capture.fields.jspace_vector:
            capture_bytes += rows * width * 2
            notes.append("full-vector capture substantially increases storage")
    total = model_bytes + lens_bytes + pinv_bytes + kv_bytes + capture_bytes
    return MemoryEstimate(
        model_gb=_bytes_to_gb(model_bytes),
        lens_gb=_bytes_to_gb(lens_bytes),
        pinv_gb=_bytes_to_gb(pinv_bytes),
        kv_cache_gb=_bytes_to_gb(kv_bytes),
        capture_gb=_bytes_to_gb(capture_bytes),
        total_gb=_bytes_to_gb(total),
        notes=notes,
    )


def memory_preflight(cfg: AppConfig, *, available_gb: float | None = None) -> MemoryEstimate:
    estimate = estimate_run_memory_gb(cfg)
    if not cfg.runtime.memory_preflight:
        return estimate
    if available_gb is None:
        return estimate
    needed = estimate.total_gb + cfg.runtime.minimum_free_memory_gb
    if needed > available_gb:
        raise MemoryError(
            f"memory preflight failed: estimated {estimate.total_gb:.1f} GiB plus "
            f"{cfg.runtime.minimum_free_memory_gb:.1f} GiB headroom exceeds "
            f"available {available_gb:.1f} GiB. Use offload, fewer capture layers, "
            "or a smaller model with a matching fitted J-Lens."
        )
    return estimate
