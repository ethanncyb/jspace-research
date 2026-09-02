from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from ..phase1.data import (
    POSITIONS,
    TASK_DISPLAY,
    balanced_quotas,
    build_messages,
    construct_target,
    hash_messages,
    normalize_context_record,
    read_jsonl,
    render_ids,
    write_jsonl_exclusive,
)
from .common import content_hash, require_generation_context, save_record
from .config import BIPIA_ATTACKS_PER_TASK

MANIFEST_NAME = "bipia_test_manifest.jsonl"
TEST_VARIANTS = (0, 1, 2, 3, 4)
SMOKE_ATTACKS = 2


def _case(
    task: str,
    builder: Any,
    record: dict[str, Any],
    context: str,
    *,
    category: str | None = None,
    variant: int | None = None,
    position: str | None = None,
    attack_text: str | None = None,
) -> dict[str, Any]:
    condition = "attack" if attack_text is not None else "control"
    clean_case_id = f"bipia:{record['context_id']}:control"
    case_id = clean_case_id
    if condition == "attack":
        case_id = f"bipia:{record['context_id']}:attack:{category}:{variant}:{position}"
    messages = build_messages(builder, record, context)
    return {
        "case_id": case_id,
        "context_id": record["context_id"],
        "source_clean_case_id": clean_case_id,
        "benchmark": "bipia",
        "task": task,
        "subgroup": TASK_DISPLAY[task],
        "condition": condition,
        "attack_category": category,
        "attack_variant_id": variant,
        "position": position,
        "attack_text": attack_text,
        "messages": messages,
        "target": construct_target(builder, record),
        "prompt_hash": hash_messages(messages),
    }


def _build_task_cases(
    *,
    task: str,
    records: Sequence[dict[str, Any]],
    attacks: dict[str, list[str]],
    builder: Any,
    insertions: dict[str, Callable[..., str]],
    attack_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not records:
        raise ValueError(f"BIPIA official test contains no contexts for task {task}")
    if not attacks or any(len(variants) != 5 for variants in attacks.values()):
        raise ValueError("Every BIPIA test attack category must contain exactly five variants")

    rng = random.Random(f"{seed}:{task}")
    cells = [(category, position) for category in sorted(attacks) for position in POSITIONS]
    rng.shuffle(cells)
    quotas = balanced_quotas(cells, attack_count)
    variants = list(TEST_VARIANTS)
    rng.shuffle(variants)
    contexts = list(records)
    rng.shuffle(contexts)

    assignments = [
        (category, position)
        for category, position in cells
        for _ in range(quotas[(category, position)])
    ]
    cases: list[dict[str, Any]] = []
    used_contexts: dict[str, dict[str, Any]] = {}
    for index, (category, position) in enumerate(assignments):
        record = contexts[index % len(contexts)]
        variant = variants[index % len(variants)]
        attack_text = attacks[category][variant]
        attacked_context = insertions[position](
            record["context"], attack_text, random_state=seed
        )
        cases.append(
            _case(
                task,
                builder,
                record,
                attacked_context,
                category=category,
                variant=variant,
                position=position,
                attack_text=attack_text,
            )
        )
        used_contexts[record["context_id"]] = record

    if len({row["prompt_hash"] for row in cases}) != len(cases):
        raise RuntimeError(f"Duplicate BIPIA attacked prompt for task {task}")
    cases.extend(
        _case(task, builder, record, record["context"])
        for _, record in sorted(used_contexts.items())
    )
    return sorted(cases, key=lambda row: row["case_id"]), {
        "categories": sorted(attacks),
        "source_context_count": len(records),
    }


def _build_manifest(config: Any) -> list[dict[str, Any]]:
    from bipia.data import AutoPIABuilder
    from bipia.data.utils import insert_end, insert_middle, insert_start

    insertions = {"start": insert_start, "middle": insert_middle, "end": insert_end}
    tasks = list(config.phase1.tasks[:1] if config.smoke else config.phase1.tasks)
    cases: list[dict[str, Any]] = []
    task_details: dict[str, Any] = {}
    for task in tasks:
        path = config.bipia_root / task / "test.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing BIPIA official test data: {path}")
        records = [
            normalize_context_record(task, raw, f"{task}:test:{index:05d}")
            for index, raw in enumerate(read_jsonl(path))
        ]
        attack_file = "code_attack_test.json" if task == "code" else "text_attack_test.json"
        with (config.bipia_root / attack_file).open("r", encoding="utf-8") as handle:
            attacks = json.load(handle)
        if not isinstance(attacks, dict):
            raise ValueError(f"Invalid BIPIA test attack file: {attack_file}")
        task_cases, task_details[task] = _build_task_cases(
            task=task,
            records=records,
            attacks=attacks,
            builder=AutoPIABuilder.from_name(task)(seed=config.phase1.seed),
            insertions=insertions,
            attack_count=SMOKE_ATTACKS if config.smoke else BIPIA_ATTACKS_PER_TASK,
            seed=config.phase1.seed,
        )
        cases.extend(task_cases)

    metadata = {
        "record_type": "metadata",
        "schema_version": 1,
        "bipia_revision": config.phase1.dependencies.bipia_revision,
        "seed": config.phase1.seed,
        "mode": "smoke" if config.smoke else "scientific",
        "attacks_per_task": BIPIA_ATTACKS_PER_TASK,
        "positions": list(POSITIONS),
        "test_variants": list(TEST_VARIANTS),
        "tasks": task_details,
    }
    return [metadata, *sorted(cases, key=lambda row: row["case_id"])]


def _validate_manifest(config: Any, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError("BIPIA test manifest is empty")
    metadata, *cases = rows
    expected = {
        "record_type": "metadata",
        "schema_version": 1,
        "bipia_revision": config.phase1.dependencies.bipia_revision,
        "seed": config.phase1.seed,
        "mode": "smoke" if config.smoke else "scientific",
        "attacks_per_task": BIPIA_ATTACKS_PER_TASK,
        "positions": list(POSITIONS),
        "test_variants": list(TEST_VARIANTS),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("BIPIA test manifest metadata changed")
    task_details = metadata.get("tasks")
    expected_tasks = set(config.phase1.tasks[:1] if config.smoke else config.phase1.tasks)
    if not isinstance(task_details, dict) or set(task_details) != expected_tasks:
        raise RuntimeError("BIPIA test manifest task coverage changed")

    case_ids = [row.get("case_id") for row in cases]
    if not cases or any(not isinstance(value, str) for value in case_ids):
        raise RuntimeError("BIPIA test manifest contains an invalid case")
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("BIPIA test manifest contains duplicate case IDs")
    for row in cases:
        if row.get("prompt_hash") != hash_messages(row.get("messages", [])):
            raise RuntimeError(f"BIPIA test manifest prompt changed: {row['case_id']}")
        if not isinstance(row.get("target"), str) or not row["target"].strip():
            raise RuntimeError(f"BIPIA test manifest target is missing: {row['case_id']}")

    attacks = [row for row in cases if row.get("condition") == "attack"]
    controls = [row for row in cases if row.get("condition") == "control"]
    if len(attacks) + len(controls) != len(cases):
        raise RuntimeError("BIPIA test manifest contains an invalid condition")
    if len({row["prompt_hash"] for row in attacks}) != len(attacks):
        raise RuntimeError("BIPIA test manifest contains duplicate attacked prompts")
    controls_by_id = {row["case_id"]: row for row in controls}
    required_controls = {row.get("source_clean_case_id") for row in attacks}
    if required_controls != set(controls_by_id) or any(
        row["context_id"] != controls_by_id[row["source_clean_case_id"]]["context_id"]
        for row in attacks
    ):
        raise RuntimeError("BIPIA test manifest attack-to-clean mapping changed")

    quota = SMOKE_ATTACKS if config.smoke else BIPIA_ATTACKS_PER_TASK
    for task, details in task_details.items():
        selected = [row for row in attacks if row["task"] == task]
        categories = details.get("categories", [])
        if len(selected) != quota or not categories:
            raise RuntimeError(f"BIPIA test attack quota changed for task {task}")
        cell_counts = Counter((row["attack_category"], row["position"]) for row in selected)
        cells = [(category, position) for category in categories for position in POSITIONS]
        counts = [cell_counts.get(cell, 0) for cell in cells]
        if set(cell_counts) - set(cells) or max(counts) - min(counts) > 1:
            raise RuntimeError(f"BIPIA category-position balance changed for task {task}")
        if not config.smoke:
            variant_counts = Counter(int(row["attack_variant_id"]) for row in selected)
            if set(variant_counts) != set(TEST_VARIANTS) or max(variant_counts.values()) - min(
                variant_counts.values()
            ) > 1:
                raise RuntimeError(f"BIPIA test variant balance changed for task {task}")
        context_counts = Counter(row["context_id"] for row in selected)
        source_count = int(details.get("source_context_count", 0))
        if len(context_counts) != min(quota, source_count) or max(
            context_counts.values()
        ) - min(context_counts.values()) > 1:
            raise RuntimeError(f"BIPIA source-context balance changed for task {task}")

    return sorted(cases, key=lambda row: row["case_id"])


def load_manifest(config: Any) -> list[dict[str, Any]]:
    path = config.output_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen BIPIA test manifest: {path}")
    return _validate_manifest(config, read_jsonl(path))


def prepare_manifest(config: Any) -> list[dict[str, Any]]:
    path = config.output_dir / MANIFEST_NAME
    if path.exists():
        return load_manifest(config)
    rows = _build_manifest(config)
    cases = _validate_manifest(config, rows)
    write_jsonl_exclusive(path, rows)
    return cases


def generate(
    config: Any,
    model: Any,
    scorer: Any,
    completed: dict[str, dict[str, Any]],
    identity: dict[str, Any],
) -> None:
    output_path = config.output_dir / "bipia_records.jsonl"
    cases = load_manifest(config)
    expected_ids = {case["case_id"] for case in cases}
    unexpected = sorted(set(completed) - expected_ids)
    if unexpected:
        raise RuntimeError(f"Unexpected cached BIPIA case ID: {unexpected[0]}")
    for case in cases:
        if case["case_id"] in completed:
            if completed[case["case_id"]].get("case_hash") != content_hash(case):
                raise RuntimeError("Cached BIPIA prompt identity changed")
            continue
        input_ids = render_ids(model.tokenizer, case["messages"])
        require_generation_context(
            int(input_ids.shape[-1]), model.context_length, config.max_new_tokens, "BIPIA"
        )
        tokens, residual = model.generate_with_capture(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            layer=scorer.mean["selected_layer"],
        )
        result = scorer.score(residual, scorer.dictionary)
        generation = model.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        save_record(
            output_path,
            {
                **identity,
                **{key: value for key, value in case.items() if key != "messages"},
                "case_hash": content_hash(case),
                **result,
                "injection_exposed": case["condition"] == "attack",
                "generated_response": generation,
                "native_valid": None,
                "native_utility": None,
                "native_attack_success": None,
            },
        )
