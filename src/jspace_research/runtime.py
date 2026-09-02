from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_parquet(path: str | Path, frame: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(path: str | Path, frame: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: str | Path, value: Any) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".pt", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_figure(path: str | Path, figure: Any, **savefig_kwargs: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=target.suffix, dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, **savefig_kwargs)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_resumable_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read an append-only JSONL cache, dropping only an incomplete final write."""

    target = Path(path)
    if not target.exists():
        return []

    rows: list[dict[str, Any]] = []
    with target.open("r+b") as handle:
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            at_end = handle.tell() == os.fstat(handle.fileno()).st_size
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if at_end and not raw_line.endswith(b"\n"):
                    handle.seek(line_start)
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                    break
                raise ValueError(f"Malformed JSONL cache record at {target}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON objects in cache: {target}")
            rows.append(value)
            if at_end and not raw_line.endswith(b"\n"):
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                break
    return rows


def ensure_cache_metadata(path: str | Path, expected: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        actual = read_json(target)
        if actual != expected:
            raise RuntimeError(f"Cache metadata mismatch at {target}. Use a new output directory.")
    else:
        atomic_write_json(target, expected)


def validate_identity_fields(
    path: str | Path, value: dict[str, Any], expected: dict[str, Any]
) -> None:
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(
                f"Cached result identity mismatch at {path}; use a new output directory"
            )


def update_provenance(
    path: str | Path,
    base: dict[str, Any],
    *,
    defaults: dict[str, Any] | None = None,
    updates: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    base = {**base, "jspace_research_git_commit": repository_git_commit()}
    if target.exists():
        value = read_json(target)
        for key, expected in base.items():
            if value.get(key) != expected:
                raise RuntimeError(
                    f"Provenance mismatch at {target}; use a new output directory"
                )
    else:
        value = {**base, **(defaults or {})}
    if updates:
        value.update(updates)
    atomic_write_json(target, value)


def repository_git_commit() -> str:
    """Return the exact repository commit used to execute an experiment stage."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Cannot determine the jspace-research Git commit; run from a Git checkout"
        ) from exc
    commit = result.stdout.strip().lower()
    if len(commit) not in (40, 64) or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("Git returned an invalid jspace-research commit")
    return commit


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def cuda_metadata(*, model_input_device: str | None) -> dict[str, Any]:
    import torch

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return {
        "device_count": len(devices),
        "devices": devices,
        "model_input_device": model_input_device,
    }
