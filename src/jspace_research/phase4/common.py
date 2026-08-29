from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from ..runtime import append_jsonl, read_resumable_jsonl, validate_identity_fields


def verify_checkout(root: Path, expected_revision: str, name: str) -> None:
    try:
        actual = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot verify the {name} checkout at {root}") from exc
    if actual != expected_revision:
        raise RuntimeError(f"{name} revision mismatch: expected {expected_revision}, found {actual}")


def content_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def completed_records(path: Path, identity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = read_resumable_jsonl(path)
    completed: dict[str, dict[str, Any]] = {}
    for row in rows:
        validate_identity_fields(path, row, identity)
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in completed:
            raise RuntimeError(f"Duplicate or invalid case ID in {path}")
        if not isinstance(row.get("case_hash"), str) or not isinstance(
            row.get("generated_response"), str
        ):
            raise RuntimeError(f"Incomplete Phase 4 record in {path}")
        detector_values = [
            row.get("mean_score"),
            row.get("mean_prediction"),
            row.get("logistic_score"),
            row.get("logistic_prediction"),
        ]
        if any(value is None for value in detector_values):
            if not all(value is None for value in detector_values):
                raise RuntimeError(f"Partially missing detector output in {path}")
        elif not (
            all(math.isfinite(float(value)) for value in detector_values[::2])
            and all(isinstance(value, bool) for value in detector_values[1::2])
        ):
            raise RuntimeError(f"Invalid detector output in {path}")
        completed[case_id] = row
    return completed


def save_record(path: Path, record: dict[str, Any]) -> None:
    append_jsonl(path, record)
