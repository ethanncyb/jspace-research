from __future__ import annotations

from pathlib import Path

import pytest

from jspace_research.runtime import (
    append_jsonl,
    atomic_write_json,
    read_json,
    read_resumable_jsonl,
    repository_git_commit,
    update_provenance,
)


def test_repository_git_commit_is_exact_revision() -> None:
    commit = repository_git_commit()
    assert len(commit) in (40, 64)
    assert set(commit) <= set("0123456789abcdef")


def test_provenance_records_code_revisions_without_treating_them_as_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "provenance.json"
    commits = iter(("a" * 40, "b" * 40))
    monkeypatch.setattr(
        "jspace_research.runtime.repository_git_commit", lambda: next(commits)
    )

    update_provenance(path, {"config_sha256": "fixed"})
    legacy = read_json(path)
    legacy.pop("jspace_research_git_commits")
    atomic_write_json(path, legacy)
    update_provenance(path, {"config_sha256": "fixed"})

    provenance = read_json(path)
    assert provenance["jspace_research_git_commit"] == "b" * 40
    assert provenance["jspace_research_git_commits"] == ["a" * 40, "b" * 40]


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
