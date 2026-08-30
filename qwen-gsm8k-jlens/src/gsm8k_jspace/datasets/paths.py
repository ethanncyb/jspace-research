"""Resolve benchmark data directories relative to the project or CWD."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_data_dir(configured: str) -> Path:
    path = Path(configured).expanduser()
    if path.is_absolute() and path.is_dir():
        return path
    candidates = [
        Path.cwd() / path,
        project_root() / path,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]
