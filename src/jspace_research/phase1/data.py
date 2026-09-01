from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cache import sha256_file
from .config import Phase1Config

TASK_DISPLAY = {
    "email": "EmailQA",
    "qa": "WebQA",
    "table": "TableQA",
    "abstract": "Summarization",
    "code": "CodeQA",
}
POSITIONS = ("start", "middle", "end")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl_exclusive(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"Frozen manifest already exists: {target}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        if target.exists():
            raise RuntimeError(f"Frozen manifest already exists: {target}")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def split_contexts(
    records: Sequence[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) < 2:
        raise ValueError("At least two source contexts are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    n_train = int(math.floor(len(records) * 2 / 3))
    n_train = min(max(n_train, 1), len(records) - 1)
    train = [records[int(index)] for index in order[:n_train]]
    validation = [records[int(index)] for index in order[n_train:]]
    return train, validation


def partition_attack_variants(
    attacks: dict[str, list[str]],
) -> tuple[dict[str, list[tuple[int, str]]], dict[str, list[tuple[int, str]]]]:
    train: dict[str, list[tuple[int, str]]] = {}
    validation: dict[str, list[tuple[int, str]]] = {}
    for category, variants in sorted(attacks.items()):
        if not isinstance(variants, list) or len(variants) < 5:
            raise ValueError(
                f"Attack category {category!r} must contain at least five variants "
                "to guarantee train/validation disjointness"
            )
        train[category] = [(index, variants[index]) for index in (0, 1, 2)]
        validation[category] = [(index, variants[index]) for index in (3, 4)]
    if not train:
        raise ValueError("The BIPIA attack file contains no categories")
    return train, validation


def balanced_quotas(keys: Sequence[tuple[str, str]], total: int) -> dict[tuple[str, str], int]:
    if not keys:
        raise ValueError("Cannot allocate quotas without category-position cells")
    quotient, remainder = divmod(total, len(keys))
    return {key: quotient + (1 if index < remainder else 0) for index, key in enumerate(keys)}


def normalize_context_record(task: str, raw: dict[str, Any], context_id: str) -> dict[str, Any]:
    def join(value: Any) -> Any:
        return "\n".join(value) if isinstance(value, list) else value

    if task == "code":
        return {
            "context_id": context_id,
            "context": join(raw["context"]),
            "code": join(raw["code"]),
            "error": join(raw["error"]),
            "ideal": join(raw["ideal"]),
        }
    if task == "abstract":
        return {
            "context_id": context_id,
            "context": raw["context"],
            "ideal": raw["ideal"],
        }
    return {
        "context_id": context_id,
        "context": raw["context"],
        "question": raw["question"],
        "ideal": raw["ideal"],
    }


def hash_messages(messages: Sequence[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_ids(tokenizer: Any, messages: Sequence[dict[str, str]]) -> Any:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return encoded["input_ids"] if isinstance(encoded, Mapping) else encoded


def prompt_length(tokenizer: Any, messages: Sequence[dict[str, str]]) -> int:
    return int(render_ids(tokenizer, messages).shape[-1])


def build_messages(
    builder: Any,
    record: dict[str, Any],
    context_text: str,
) -> list[dict[str, str]]:
    example = dict(record)
    example["context"] = context_text
    system_prompt, user_prompt = builder.construct_prompt(example, require_system_prompt=True)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def construct_target(builder: Any, record: dict[str, Any]) -> str:
    target = builder.construct_response(record)
    if not isinstance(target, str) or not target.strip():
        raise ValueError("BIPIA construct_response returned an invalid target")
    return target


def make_benign_match(
    *,
    tokenizer: Any,
    builder: Any,
    base_record: dict[str, Any],
    position: str,
    target_length: int,
    donors: Sequence[dict[str, Any]],
    rng: random.Random,
    tolerance: int,
    seed: int,
    insert_fn: Callable[[str, str, int | None], str],
) -> tuple[str, list[dict[str, str]], int] | None:
    donor_order = list(range(len(donors)))
    rng.shuffle(donor_order)
    for donor_index in donor_order[:30]:
        donor = donors[donor_index]
        if donor["context_id"] == base_record["context_id"]:
            continue
        donor_ids = tokenizer.encode(donor["context"], add_special_tokens=False)
        if len(donor_ids) < 8:
            continue

        low, high = 1, len(donor_ids)
        best: tuple[int, int] | None = None
        while low <= high:
            middle = (low + high) // 2
            span = tokenizer.decode(donor_ids[:middle], skip_special_tokens=True)
            control_context = insert_fn(base_record["context"], span, random_state=seed)
            messages = build_messages(builder, base_record, control_context)
            length = prompt_length(tokenizer, messages)
            if best is None or abs(length - target_length) < abs(best[0] - target_length):
                best = (length, middle)
            if length < target_length:
                low = middle + 1
            elif length > target_length:
                high = middle - 1
            else:
                break

        if best is None:
            continue
        center = best[1]
        for count in range(max(1, center - 16), min(len(donor_ids), center + 16) + 1):
            span = tokenizer.decode(donor_ids[:count], skip_special_tokens=True)
            control_context = insert_fn(base_record["context"], span, random_state=seed)
            messages = build_messages(builder, base_record, control_context)
            length = prompt_length(tokenizer, messages)
            if abs(length - target_length) <= tolerance:
                return span, messages, length
    return None


def _task_context_path(config: Phase1Config, task: str) -> Path:
    if task == "qa":
        assert config.data.webqa_train_path is not None
        return config.data.webqa_train_path
    if task == "abstract":
        assert config.data.summarization_train_path is not None
        return config.data.summarization_train_path
    return config.data.bipia_root / task / "train.jsonl"


def _load_attacks(config: Phase1Config, task: str) -> dict[str, list[str]]:
    filename = "code_attack_train.json" if task == "code" else "text_attack_train.json"
    with (config.data.bipia_root / filename).open("r", encoding="utf-8") as handle:
        attacks = json.load(handle)
    if not isinstance(attacks, dict):
        raise ValueError(f"BIPIA attacks must be a mapping: {filename}")
    return attacks


def _load_bipia_helpers() -> tuple[Any, dict[str, Callable[..., str]]]:
    try:
        from bipia.data import AutoPIABuilder
        from bipia.data.utils import insert_end, insert_middle, insert_start
    except ImportError as exc:
        raise RuntimeError(
            "BIPIA is not importable. Install the pinned checkout before running prepare."
        ) from exc
    return AutoPIABuilder, {
        "start": insert_start,
        "middle": insert_middle,
        "end": insert_end,
    }


def verify_bipia_revision(config: Phase1Config) -> None:
    checkout = config.data.bipia_root.parent
    try:
        actual = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot verify the BIPIA checkout at {checkout}") from exc
    expected = config.dependencies.bipia_revision
    if actual != expected:
        raise RuntimeError(f"BIPIA revision mismatch: expected {expected}, found {actual}")
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)


def build_pairs_for_task(
    config: Phase1Config,
    task: str,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    AutoPIABuilder, insert_fns = _load_bipia_helpers()
    path = _task_context_path(config, task)
    raw_contexts = read_jsonl(path)
    records = [
        normalize_context_record(task, raw, f"{task}:{index}")
        for index, raw in enumerate(raw_contexts)
    ]
    train_contexts, validation_contexts = split_contexts(records, config.seed)
    train_attacks, validation_attacks = partition_attack_variants(_load_attacks(config, task))
    builder = AutoPIABuilder.from_name(task)(seed=config.seed)

    rows: list[dict[str, Any]] = []
    seen_attack_hashes: set[str] = set()
    seen_control_hashes: set[str] = set()
    specifications = (
        ("train", train_contexts, train_attacks, config.train_pairs_per_task, config.seed),
        (
            "validation",
            validation_contexts,
            validation_attacks,
            config.validation_pairs_per_task,
            config.seed + 10_000,
        ),
    )

    for split, contexts, attack_map, total, split_seed in specifications:
        rng = random.Random(split_seed)
        cells = [(category, position) for category in sorted(attack_map) for position in POSITIONS]
        quotas = balanced_quotas(cells, total)
        split_index = 0
        for category, position in cells:
            target = quotas[(category, position)]
            made = 0
            attempts = 0
            while made < target and attempts < max(200, target * 200):
                attempts += 1
                context = rng.choice(contexts)
                variant_id, attack_text = rng.choice(attack_map[category])
                attack_context = insert_fns[position](
                    context["context"], attack_text, random_state=config.seed
                )
                attack_messages = build_messages(builder, context, attack_context)
                attack_length = prompt_length(tokenizer, attack_messages)
                attack_hash = hash_messages(attack_messages)
                if attack_length > config.max_input_tokens or attack_hash in seen_attack_hashes:
                    continue

                control = make_benign_match(
                    tokenizer=tokenizer,
                    builder=builder,
                    base_record=context,
                    position=position,
                    target_length=attack_length,
                    donors=contexts,
                    rng=rng,
                    tolerance=config.token_match_tolerance,
                    seed=config.seed,
                    insert_fn=insert_fns[position],
                )
                if control is None:
                    continue
                benign_text, control_messages, control_length = control
                control_hash = hash_messages(control_messages)
                if control_length > config.max_input_tokens or control_hash in seen_control_hashes:
                    continue

                rows.append(
                    {
                        "pair_id": f"{task}:{split}:{split_index:05d}",
                        "task": task,
                        "task_display": TASK_DISPLAY[task],
                        "split": split,
                        "context_id": context["context_id"],
                        "attack_category": category,
                        "attack_variant_id": int(variant_id),
                        "position": position,
                        "attack_text": attack_text,
                        "benign_text": benign_text,
                        "target": construct_target(builder, context),
                        "attack_messages": attack_messages,
                        "control_messages": control_messages,
                        "attack_prompt_tokens": attack_length,
                        "control_prompt_tokens": control_length,
                        "attack_prompt_hash": attack_hash,
                        "control_prompt_hash": control_hash,
                    }
                )
                seen_attack_hashes.add(attack_hash)
                seen_control_hashes.add(control_hash)
                split_index += 1
                made += 1
            if made != target:
                raise RuntimeError(
                    f"Built {made}/{target} pairs for {task}/{split}/{category}/{position}; "
                    "no manifest was frozen"
                )
    return rows


def validate_pair_manifest(rows: Sequence[dict[str, Any]], config: Phase1Config) -> None:
    expected = {(task, "train"): config.train_pairs_per_task for task in config.tasks}
    expected.update(
        {(task, "validation"): config.validation_pairs_per_task for task in config.tasks}
    )
    counts: dict[tuple[str, str], int] = {}
    pair_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    contexts: dict[tuple[str, str], set[str]] = {}
    variants: dict[tuple[str, str], set[tuple[str, int]]] = {}
    categories: dict[tuple[str, str], set[str]] = {}
    cell_counts: dict[tuple[str, str, str, str], int] = {}

    for row in rows:
        key = (row["task"], row["split"])
        counts[key] = counts.get(key, 0) + 1
        if row["pair_id"] in pair_ids:
            raise ValueError(f"Duplicate pair_id: {row['pair_id']}")
        pair_ids.add(row["pair_id"])
        for hash_key in ("attack_prompt_hash", "control_prompt_hash"):
            prompt_hash = row[hash_key]
            if prompt_hash in prompt_hashes:
                raise ValueError(f"Duplicate prompt hash: {prompt_hash}")
            prompt_hashes.add(prompt_hash)
        if (
            abs(row["attack_prompt_tokens"] - row["control_prompt_tokens"])
            > config.token_match_tolerance
        ):
            raise ValueError(f"Token-length mismatch for {row['pair_id']}")
        if max(row["attack_prompt_tokens"], row["control_prompt_tokens"]) > config.max_input_tokens:
            raise ValueError(f"Overlength prompt for {row['pair_id']}")
        if row["position"] not in POSITIONS:
            raise ValueError(f"Invalid insertion position for {row['pair_id']}")
        if not isinstance(row.get("target"), str) or not row["target"].strip():
            raise ValueError(f"Missing BIPIA construct_response target for {row['pair_id']}")
        contexts.setdefault(key, set()).add(row["context_id"])
        category = row["attack_category"]
        variant_id = int(row["attack_variant_id"])
        allowed_variants = {0, 1, 2} if row["split"] == "train" else {3, 4}
        if variant_id not in allowed_variants:
            raise ValueError(f"Invalid {row['split']} attack variant for {row['pair_id']}")
        variants.setdefault(key, set()).add((category, variant_id))
        categories.setdefault(key, set()).add(category)
        cell = (
            row["task"],
            row["split"],
            category,
            row["position"],
        )
        cell_counts[cell] = cell_counts.get(cell, 0) + 1

    if counts != expected:
        raise ValueError(f"Pair quotas do not match: expected {expected}, found {counts}")
    for task in config.tasks:
        if not contexts[(task, "train")].isdisjoint(contexts[(task, "validation")]):
            raise ValueError(f"Source-context leakage for task {task}")
        if not variants[(task, "train")].isdisjoint(variants[(task, "validation")]):
            raise ValueError(f"Attack-variant leakage for task {task}")
        task_categories = sorted(categories[(task, "train")] | categories[(task, "validation")])
        for split in ("train", "validation"):
            cells = [(category, position) for category in task_categories for position in POSITIONS]
            quotas = balanced_quotas(cells, expected[(task, split)])
            actual = {cell: cell_counts.get((task, split, cell[0], cell[1]), 0) for cell in cells}
            if actual != quotas:
                raise ValueError(f"Unbalanced category-position cells for {task}/{split}")


def prepare_manifest(config: Phase1Config, tokenizer: Any) -> tuple[Path, list[dict[str, Any]]]:
    config.validate(require_data_files=True)
    verify_bipia_revision(config)
    manifest_path = config.output_dir / "pair_manifest.jsonl"
    if manifest_path.exists():
        rows = read_jsonl(manifest_path)
        validate_pair_manifest(rows, config)
        return manifest_path, rows

    rows: list[dict[str, Any]] = []
    for task in config.tasks:
        rows.extend(build_pairs_for_task(config, task, tokenizer))
    validate_pair_manifest(rows, config)
    write_jsonl_exclusive(manifest_path, rows)
    return manifest_path, rows


def expand_examples(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        common = {
            "pair_id": row["pair_id"],
            "task": row["task"],
            "task_display": row["task_display"],
            "split": row["split"],
            "context_id": row["context_id"],
            "attack_category": row["attack_category"],
            "attack_variant_id": row["attack_variant_id"],
            "position": row["position"],
            "attack_text": row["attack_text"],
            "target": row["target"],
        }
        examples.append(
            {
                **common,
                "condition": "attack",
                "label": 1,
                "messages": row["attack_messages"],
                "prompt_hash": row["attack_prompt_hash"],
            }
        )
        examples.append(
            {
                **common,
                "condition": "control",
                "label": 0,
                "messages": row["control_messages"],
                "prompt_hash": row["control_prompt_hash"],
            }
        )
    for index, example in enumerate(examples):
        example["example_index"] = index
    return examples


def manifest_sha256(path: str | Path) -> str:
    return sha256_file(path)
