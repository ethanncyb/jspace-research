"""Repoint a Phase 4 run at re-saved Phase 3 detector files with unchanged contents.

Re-running Phase 3 before detector saves were byte-stable rewrote the detector files
with new hashes even when every value was identical. Phase 4 keys its cached records
and provenance on those file hashes. This migration updates the recorded hashes only
when every other provenance field, including both thresholds and the logistic feature
count, still matches, and it logs the change in provenance.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..runtime import atomic_write_json, read_json, read_resumable_jsonl, repository_git_commit
from .config import load_config

HASH_FIELDS = ("mean_detector_sha256", "logistic_detector_sha256")
DETECTOR_HASH_FIELDS = ("mean_sha256", "logistic_sha256")
MIGRATED_RECORDS = ("bipia_records.jsonl", "injecagent_records.jsonl")
BACKUP_SUFFIX = ".before_detector_rehash"


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def detector_hash_changes(saved: dict[str, Any], current: dict[str, Any]) -> dict[str, dict]:
    """Return old and new detector hashes, refusing any other provenance difference."""

    saved = _normalized(saved)
    current = _normalized(current)
    for field, detector_field in zip(HASH_FIELDS, DETECTOR_HASH_FIELDS, strict=True):
        if (saved.get("detectors") or {}).get(detector_field) != saved.get(field):
            raise RuntimeError(f"Saved provenance has inconsistent {field} values")
    ignored = set(HASH_FIELDS) | {"detectors"}
    differing = sorted(
        key for key in current if key not in ignored and saved.get(key) != current[key]
    )
    saved_detectors = dict(saved.get("detectors") or {})
    current_detectors = dict(current["detectors"])
    for key in DETECTOR_HASH_FIELDS:
        saved_detectors.pop(key, None)
        current_detectors.pop(key, None)
    if saved_detectors != current_detectors:
        differing.append("detectors (thresholds or feature count)")
    if differing:
        raise RuntimeError(
            "Phase 4 provenance differs beyond detector file hashes: " + ", ".join(differing)
        )
    changes = {
        field: {"old": saved.get(field), "new": current[field]}
        for field in HASH_FIELDS
        if saved.get(field) != current[field]
    }
    return changes


def _rewrite_records(path: Path, changes: dict[str, dict]) -> int:
    rows = read_resumable_jsonl(path)
    for row in rows:
        for field, change in changes.items():
            if row.get(field) != change["old"]:
                raise RuntimeError(f"Unexpected {field} in {path}: {row.get(field)!r}")
            row[field] = change["new"]
    shutil.copy2(path, path.with_name(path.name + BACKUP_SUFFIX))
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return len(rows)


def rehash(config: Any, *, apply: bool) -> dict[str, dict]:
    from .pipeline import _base_provenance, _handoff

    provenance_path = config.output_dir / "provenance.json"
    saved = read_json(provenance_path)
    handoff, detectors = _handoff(config)
    current = _base_provenance(config, handoff, detectors)
    changes = detector_hash_changes(saved, current)
    if not changes:
        print("Detector hashes already match; nothing to migrate.")
        return changes
    for field, change in changes.items():
        print(f"{field}: {change['old']} -> {change['new']}")
    stale = config.output_dir / "agentdojo_records.jsonl"
    if stale.exists():
        raise RuntimeError(
            f"Delete {stale} first; AgentDojo records from the previous harness are regenerated"
        )
    if not apply:
        print("Dry run only; rerun with --apply to update the records and provenance.")
        return changes

    for name in MIGRATED_RECORDS:
        path = config.output_dir / name
        if path.exists():
            print(f"Updated {_rewrite_records(path, changes)} rows in {path}")
    shutil.copy2(provenance_path, provenance_path.with_name(provenance_path.name + BACKUP_SUFFIX))
    saved = read_json(provenance_path)
    for field, change in changes.items():
        saved[field] = change["new"]
    saved["detectors"] = {**saved["detectors"], **current["detectors"]}
    saved.setdefault("detector_rehash_log", []).append(
        {
            "changed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "jspace_research_git_commit": repository_git_commit(),
            "changes": changes,
            "reason": (
                "Phase 3 re-saved byte-different detector files with unchanged thresholds, "
                "feature count, and Phase 1 identity"
            ),
        }
    )
    atomic_write_json(provenance_path, saved)
    print(f"Updated {provenance_path}; backups end in {BACKUP_SUFFIX}")
    return changes


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m jspace_research.phase4.rehash", description=__doc__.splitlines()[0]
    )
    for name in (
        "--config",
        "--phase1",
        "--phase3",
        "--bipia-root",
        "--agentdojo-root",
        "--injecagent-root",
        "--output-dir",
    ):
        parser.add_argument(name, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    rehash(
        load_config(
            args.config,
            phase1_selected_path=args.phase1,
            phase3_dir=args.phase3,
            bipia_root=args.bipia_root,
            agentdojo_root=args.agentdojo_root,
            injecagent_root=args.injecagent_root,
            output_dir=args.output_dir,
        ),
        apply=args.apply,
    )


if __name__ == "__main__":
    main()
