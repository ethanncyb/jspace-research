from __future__ import annotations

from pathlib import Path

import pytest

from jspace_research.runtime import (
    append_jsonl,
    read_resumable_jsonl,
    repository_git_commit,
)


def test_repository_git_commit_is_exact_revision() -> None:
    commit = repository_git_commit()
    assert len(commit) in (40, 64)
    assert set(commit) <= set("0123456789abcdef")


def test_jsonl_cache_repairs_only_an_incomplete_final_record(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    append_jsonl(path, {"job_id": "one"})
    append_jsonl(path, {"job_id": "two"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"job_id":')

    assert read_resumable_jsonl(path) == [{"job_id": "one"}, {"job_id": "two"}]
    append_jsonl(path, {"job_id": "three"})
    assert [row["job_id"] for row in read_resumable_jsonl(path)] == [
        "one",
        "two",
        "three",
    ]


def test_jsonl_cache_rejects_malformed_complete_records(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text('{"job_id":"one"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSONL"):
        read_resumable_jsonl(path)
