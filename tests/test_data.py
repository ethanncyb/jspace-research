from __future__ import annotations

from collections import UserDict
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from jspace_research.phase1.config import (
    DataConfig,
    DependencyConfig,
    LensConfig,
    ModelConfig,
    Phase1Config,
)
from jspace_research.phase1.data import (
    balanced_quotas,
    construct_target,
    partition_attack_variants,
    read_jsonl,
    render_ids,
    split_contexts,
    validate_pair_manifest,
    write_jsonl_exclusive,
)


class MappingTokenizer:
    def apply_chat_template(self, *args: object, **kwargs: object) -> UserDict:
        return UserDict({"input_ids": torch.tensor([[1, 2, 3]])})


def test_render_ids_unwraps_mapping_tokenizer_output() -> None:
    input_ids = render_ids(MappingTokenizer(), [{"role": "user", "content": "test"}])
    assert input_ids.shape == (1, 3)


def test_render_ids_disables_thinking_for_qwen_templates() -> None:
    class QwenTokenizer:
        chat_template = "{% if enable_thinking %}think{% endif %}"

        def apply_chat_template(self, *args: object, **kwargs: object) -> torch.Tensor:
            assert kwargs.get("enable_thinking") is False
            return torch.tensor([[1, 2, 3]])

    input_ids = render_ids(QwenTokenizer(), [{"role": "user", "content": "test"}])
    assert input_ids.shape == (1, 3)


def test_construct_target_uses_bipia_builder_response() -> None:
    class Builder:
        def construct_response(self, record: dict) -> str:
            return f"Answer: {record['ideal']}."

    assert construct_target(Builder(), {"ideal": "expected"}) == "Answer: expected."


def make_config(tmp_path: Path) -> Phase1Config:
    return Phase1Config(
        model=ModelConfig("model", "a" * 40),
        lens=LensConfig("lens", "b" * 40, "lens.pt", "c" * 64),
        dependencies=DependencyConfig(
            "581d398613e5602a5af361e1c34d3a92ea82ba8e",
            "a004b69ec0dd446e0afd461d98cb5e96e120a5d0",
        ),
        data=DataConfig(tmp_path / "BIPIA" / "benchmark"),
        output_dir=tmp_path / "output",
        seed=42,
        tasks=("email",),
        train_pairs_per_task=1,
        validation_pairs_per_task=1,
        max_input_tokens=4096,
        token_match_tolerance=1,
        sparsity_k=25,
        screen_candidates=512,
        decomposition_batch_size=8,
        dictionary_chunk_size=4096,
    )


def row(split: str, context: str, variant: int, suffix: str) -> dict:
    return {
        "pair_id": f"email:{split}:00000",
        "task": "email",
        "split": split,
        "context_id": context,
        "attack_category": "direct",
        "attack_variant_id": variant,
        "position": "start",
        "target": "Answer: expected.",
        "attack_prompt_hash": f"attack-{suffix}",
        "control_prompt_hash": f"control-{suffix}",
        "attack_prompt_tokens": 100,
        "control_prompt_tokens": 101,
    }


def test_split_contexts_is_deterministic_and_disjoint() -> None:
    records = [{"context_id": str(index)} for index in range(12)]
    train_a, validation_a = split_contexts(records, 42)
    train_b, validation_b = split_contexts(records, 42)
    assert train_a == train_b
    assert validation_a == validation_b
    assert {item["context_id"] for item in train_a}.isdisjoint(
        {item["context_id"] for item in validation_a}
    )


def test_attack_variants_are_strictly_disjoint() -> None:
    train, validation = partition_attack_variants(
        {"direct": ["zero", "one", "two", "three", "four"]}
    )
    assert {index for index, _ in train["direct"]} == {0, 1, 2}
    assert {index for index, _ in validation["direct"]} == {3, 4}
    with pytest.raises(ValueError, match="at least five"):
        partition_attack_variants({"direct": ["zero", "one", "two", "three"]})


def test_balanced_quotas_differ_by_at_most_one() -> None:
    keys = [("a", "start"), ("a", "middle"), ("a", "end")]
    quotas = balanced_quotas(keys, 8)
    assert sum(quotas.values()) == 8
    assert max(quotas.values()) - min(quotas.values()) == 1


def test_manifest_invariants_and_duplicate_rejection(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    rows = [row("train", "ctx-train", 0, "train"), row("validation", "ctx-val", 3, "val")]
    validate_pair_manifest(rows, config)

    leaked = [rows[0], {**rows[1], "attack_variant_id": 0}]
    with pytest.raises(ValueError, match="Invalid validation attack variant"):
        validate_pair_manifest(leaked, config)

    duplicate = [rows[0], {**rows[1], "attack_prompt_hash": "attack-train"}]
    with pytest.raises(ValueError, match="Duplicate prompt hash"):
        validate_pair_manifest(duplicate, config)

    mismatch_config = replace(config, token_match_tolerance=0)
    with pytest.raises(ValueError, match="Token-length mismatch"):
        validate_pair_manifest(rows, mismatch_config)

    overlength_config = replace(config, max_input_tokens=100)
    with pytest.raises(ValueError, match="Overlength prompt"):
        validate_pair_manifest(rows, overlength_config)

    incomplete = [rows[0], {**rows[1], "target": ""}]
    with pytest.raises(ValueError, match="construct_response target"):
        validate_pair_manifest(incomplete, config)


def test_partial_task_validation_uses_only_current_task_quota(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), tasks=("email", "qa"))
    rows = [row("train", "ctx-train", 0, "train"), row("validation", "ctx-val", 3, "val")]

    validate_pair_manifest(rows, config, tasks=("email",))
    with pytest.raises(ValueError, match="Pair quotas do not match"):
        validate_pair_manifest(rows, config)


def test_manifest_rejects_unbalanced_category_position_cells(tmp_path: Path) -> None:
    config = replace(
        make_config(tmp_path),
        train_pairs_per_task=3,
        validation_pairs_per_task=3,
    )
    rows = []
    for split, context, variant in (("train", "ctx-train", 0), ("validation", "ctx-val", 3)):
        for index in range(3):
            rows.append(
                {
                    **row(split, context, variant, f"{split}-{index}"),
                    "pair_id": f"email:{split}:{index:05d}",
                    "position": "start",
                }
            )
    with pytest.raises(ValueError, match="Unbalanced category-position"):
        validate_pair_manifest(rows, config)


def test_manifest_allows_zero_quota_categories_in_smoke_validation(tmp_path: Path) -> None:
    config = replace(
        make_config(tmp_path),
        train_pairs_per_task=12,
        validation_pairs_per_task=6,
    )
    rows = []
    split_specs = (
        ("train", "ctx-train", 0, ("a", "b", "c", "d")),
        ("validation", "ctx-val", 3, ("a", "b")),
    )
    for split, context, variant, categories in split_specs:
        index = 0
        for category in categories:
            for position in ("start", "middle", "end"):
                rows.append(
                    {
                        **row(split, context, variant, f"{split}-{index}"),
                        "pair_id": f"email:{split}:{index:05d}",
                        "attack_category": category,
                        "position": position,
                    }
                )
                index += 1

    validate_pair_manifest(rows, config)


def test_manifest_is_written_exclusively(tmp_path: Path) -> None:
    path = tmp_path / "pair_manifest.jsonl"
    rows = [{"pair_id": "one"}]
    write_jsonl_exclusive(path, rows)
    assert read_jsonl(path) == rows
    with pytest.raises(RuntimeError, match="already exists"):
        write_jsonl_exclusive(path, rows)


def test_manifest_write_does_not_require_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_hard_link(*_args: object) -> None:
        raise PermissionError("hard links are unsupported")

    monkeypatch.setattr("os.link", reject_hard_link)
    path = tmp_path / "pair_manifest.jsonl"
    write_jsonl_exclusive(path, [{"pair_id": "drive-compatible"}])
    assert read_jsonl(path) == [{"pair_id": "drive-compatible"}]
