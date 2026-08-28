from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime import (
    atomic_write_json,
    ensure_cache_metadata,
    read_json,
    sha256_file,
)

__all__ = [
    "atomic_write_json",
    "ensure_cache_metadata",
    "read_json",
    "sha256_file",
    "load_done",
    "save_done",
    "open_memmap",
    "open_uint16_memmap",
]


def load_done(path: str | Path, size: int) -> np.ndarray:
    target = Path(path)
    if not target.exists():
        return np.zeros(size, dtype=bool)
    done = np.load(target, allow_pickle=False)
    if done.dtype != np.bool_ or done.shape != (size,):
        raise RuntimeError(f"Invalid completion bitmap at {target}")
    return done


def save_done(path: str | Path, done: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
        np.save(handle, done, allow_pickle=False)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def open_memmap(
    path: str | Path,
    shape: tuple[int, ...],
    *,
    dtype: Any,
    fill_value: int | float | None = None,
) -> np.memmap:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    mode = "r+" if existed else "w+"
    numpy_dtype = np.dtype(dtype)
    expected_bytes = int(np.prod(shape)) * numpy_dtype.itemsize
    if existed and target.stat().st_size != expected_bytes:
        raise RuntimeError(f"Cache shape mismatch for {target}")
    memory_map = np.memmap(target, dtype=numpy_dtype, mode=mode, shape=shape)
    if not existed and fill_value is not None:
        memory_map.fill(fill_value)
        memory_map.flush()
    return memory_map


def open_uint16_memmap(path: str | Path, shape: tuple[int, ...]) -> np.memmap:
    return open_memmap(path, shape, dtype=np.uint16)
