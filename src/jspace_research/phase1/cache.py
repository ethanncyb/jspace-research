from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def ensure_cache_metadata(path: str | Path, expected: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        actual = read_json(target)
        if actual != expected:
            raise RuntimeError(f"Cache metadata mismatch at {target}. Use a new output directory.")
    else:
        atomic_write_json(target, expected)


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


def open_uint16_memmap(path: str | Path, shape: tuple[int, ...]) -> np.memmap:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+" if target.exists() else "w+"
    expected_bytes = int(np.prod(shape)) * np.dtype(np.uint16).itemsize
    if target.exists() and target.stat().st_size != expected_bytes:
        raise RuntimeError(f"Cache shape mismatch for {target}")
    return np.memmap(target, dtype=np.uint16, mode=mode, shape=shape)
