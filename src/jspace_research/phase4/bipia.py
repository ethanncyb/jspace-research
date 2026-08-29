from __future__ import annotations

import json
from typing import Any

from ..phase1.data import (
    POSITIONS,
    TASK_DISPLAY,
    build_messages,
    construct_target,
    hash_messages,
    normalize_context_record,
    read_jsonl,
    render_ids,
)
from .common import content_hash, save_record


def build_cases(config: Any) -> list[dict[str, Any]]:
    from bipia.data import AutoPIABuilder
    from bipia.data.utils import insert_end, insert_middle, insert_start

    insertions = {"start": insert_start, "middle": insert_middle, "end": insert_end}
    cases: list[dict[str, Any]] = []
    for task in config.phase1.tasks:
        context_path = config.bipia_root / task / "test.jsonl"
        if not context_path.is_file():
            raise FileNotFoundError(f"Missing BIPIA official test data: {context_path}")
        records = [
            normalize_context_record(task, raw, f"{task}:test:{index:05d}")
            for index, raw in enumerate(read_jsonl(context_path))
        ]
        attack_file = "code_attack_test.json" if task == "code" else "text_attack_test.json"
        with (config.bipia_root / attack_file).open("r", encoding="utf-8") as handle:
            attacks = json.load(handle)
        builder = AutoPIABuilder.from_name(task)(seed=config.phase1.seed)

        for record in records:
            clean_messages = build_messages(builder, record, record["context"])
            cases.append(
                {
                    "case_id": f"bipia:{record['context_id']}:control",
                    "benchmark": "bipia",
                    "task": task,
                    "subgroup": TASK_DISPLAY[task],
                    "condition": "control",
                    "attack_text": None,
                    "messages": clean_messages,
                    "target": construct_target(builder, record),
                    "prompt_hash": hash_messages(clean_messages),
                }
            )
            for category in sorted(attacks):
                for variant, attack_text in enumerate(attacks[category]):
                    for position in POSITIONS:
                        attacked_context = insertions[position](
                            record["context"], attack_text, random_state=config.phase1.seed
                        )
                        messages = build_messages(builder, record, attacked_context)
                        cases.append(
                            {
                                "case_id": (
                                    f"bipia:{record['context_id']}:attack:"
                                    f"{category}:{variant}:{position}"
                                ),
                                "benchmark": "bipia",
                                "task": task,
                                "subgroup": TASK_DISPLAY[task],
                                "condition": "attack",
                                "attack_category": category,
                                "attack_variant_id": variant,
                                "position": position,
                                "attack_text": attack_text,
                                "messages": messages,
                                "target": construct_target(builder, record),
                                "prompt_hash": hash_messages(messages),
                            }
                        )

    cases.sort(key=lambda row: row["case_id"])
    if config.smoke:
        attacks = [case for case in cases if case["condition"] == "attack"][:2]
        controls_by_context = {
            case["case_id"].removesuffix(":control"): case
            for case in cases
            if case["condition"] == "control"
        }
        controls = []
        for attack in attacks:
            context_id = attack["case_id"].split(":attack:")[0]
            control = dict(controls_by_context[context_id])
            control["case_id"] = attack["case_id"] + ":matched-control"
            controls.append(control)
        cases = sorted([*attacks, *controls], key=lambda row: row["case_id"])
    return cases


def generate(
    config: Any,
    model: Any,
    scorer: Any,
    completed: dict[str, dict[str, Any]],
    identity: dict[str, Any],
) -> None:
    output_path = config.output_dir / "bipia_records.jsonl"
    cases = build_cases(config)
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
        if input_ids.shape[-1] > config.phase1.max_input_tokens:
            raise RuntimeError(f"BIPIA prompt exceeds the frozen token limit: {case['case_id']}")
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
