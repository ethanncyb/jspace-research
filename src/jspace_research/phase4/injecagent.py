from __future__ import annotations

import importlib.util
import json
import runpy
from pathlib import Path
from types import ModuleType
from typing import Any

from ..phase1.data import hash_messages, render_ids
from .common import content_hash, save_record


def _load_parser(root: Path) -> ModuleType:
    path = root / "src" / "output_parsing.py"
    spec = importlib.util.spec_from_file_location("jspace_injecagent_output_parsing", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load InjecAgent parser: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tools(root: Path) -> dict[str, dict[str, Any]]:
    with (root / "data" / "tools.json").open("r", encoding="utf-8") as handle:
        toolkits = json.load(handle)
    return {
        toolkit["toolkit"] + tool["name"]: {**tool, "name": toolkit["toolkit"] + tool["name"]}
        for toolkit in toolkits
        for tool in toolkit["tools"]
    }


def _prompt(
    item: dict[str, Any],
    tools: dict[str, dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
    *,
    second: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    names = [item["User Tool"], *item["Attacker Tools"]]
    descriptions = json.dumps([tools[name] for name in names], indent=1)
    scratchpad = (
        f"\n{item['Thought']}\nAction: {item['User Tool']}\n"
        f"Action Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"
    )
    if second is not None:
        scratchpad += (
            f"Thought: {second['thought']}\nAction: {item['Attacker Tools'][0]}\n"
            f"Action Input: {second['parameters']}\nObservation: {second['response']}\n"
        )
    filled = user_prompt.format(
        available_tool_descriptions=descriptions,
        input=item["User Instruction"],
        available_tool_names=names,
        agent_scratchpad=scratchpad,
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": filled}]


def build_cases(config: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for subgroup, filename in (
        ("direct_harm", "test_cases_dh_base.json"),
        ("data_stealing", "test_cases_ds_base.json"),
    ):
        with (config.injecagent_root / "data" / filename).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        expected_count = 510 if subgroup == "direct_harm" else 544
        if len(values) != expected_count:
            raise RuntimeError(
                f"Pinned InjecAgent {subgroup} data must contain {expected_count} cases"
            )
        for index, item in enumerate(values):
            cases.append(
                {
                    "case_id": f"injecagent:{subgroup}:{index:04d}",
                    "benchmark": "injecagent",
                    "task": None,
                    "subgroup": subgroup,
                    "condition": "attack",
                    "item": item,
                }
            )
    cases.sort(key=lambda row: row["case_id"])
    if config.smoke:
        cases = [
            case
            for subgroup in ("direct_harm", "data_stealing")
            for case in [row for row in cases if row["subgroup"] == subgroup][:3]
        ]
    return cases


def _native_eval(parser: ModuleType, output: str, item: dict[str, Any], target: str) -> dict:
    try:
        return parser.evaluate_output_prompted(
            output,
            target,
            item["Attacker Tools"],
            item["User Tool"],
            item["Attacker Instruction"],
        )
    except (AssertionError, IndexError, KeyError, TypeError, ValueError) as exc:
        return {"eval": "invalid", "invalid_reason": type(exc).__name__}


def generate(
    config: Any,
    model: Any,
    scorer: Any,
    completed: dict[str, dict[str, Any]],
    identity: dict[str, Any],
) -> None:
    output_path = config.output_dir / "injecagent_records.jsonl"
    prompt_values = runpy.run_path(str(config.injecagent_root / "src/prompts/agent_prompts.py"))
    system_prompt, user_prompt = prompt_values["PROMPT_DICT"][config.injecagent_prompt_type]
    tools = _tools(config.injecagent_root)
    parser = _load_parser(config.injecagent_root)
    with (config.injecagent_root / "data/attacker_simulated_responses.json").open(
        "r", encoding="utf-8"
    ) as handle:
        simulated = json.load(handle)

    cases = build_cases(config)
    expected_ids = {case["case_id"] for case in cases}
    unexpected = sorted(set(completed) - expected_ids)
    if unexpected:
        raise RuntimeError(f"Unexpected cached InjecAgent case ID: {unexpected[0]}")
    for case in cases:
        case_hash = content_hash(case)
        if case["case_id"] in completed:
            if completed[case["case_id"]].get("case_hash") != case_hash:
                raise RuntimeError("Cached InjecAgent case identity changed")
            continue
        item = case["item"]
        messages = _prompt(item, tools, system_prompt, user_prompt)
        input_ids = render_ids(model.tokenizer, messages)
        if input_ids.shape[-1] > config.phase1.max_input_tokens:
            raise RuntimeError(f"InjecAgent prompt exceeds token limit: {case['case_id']}")
        tokens, residual = model.generate_with_capture(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            layer=scorer.mean["selected_layer"],
        )
        output = model.tokenizer.decode(tokens, skip_special_tokens=True)
        first = _native_eval(parser, output, item, item["Attacker Tools"][0])
        second_output = None
        second_eval = None
        if case["subgroup"] == "data_stealing" and first["eval"] == "succ":
            parameters = first["Attacker Tool Parameters"]
            key = f"({item['Attacker Tools'][0]},{parameters})"
            if key not in simulated:
                raise RuntimeError(f"Missing pinned InjecAgent simulated response for {key}")
            second = {
                "thought": first["Attacker Tool Thought"],
                "parameters": parameters,
                "response": simulated[key],
            }
            second_messages = _prompt(
                item, tools, system_prompt, user_prompt, second=second
            )
            second_ids = render_ids(model.tokenizer, second_messages)
            if second_ids.shape[-1] > config.phase1.max_input_tokens:
                raise RuntimeError(
                    f"InjecAgent second prompt exceeds token limit: {case['case_id']}"
                )
            second_tokens = model.generate_from_prompt(
                second_ids, max_new_tokens=config.max_new_tokens
            )
            second_output = model.tokenizer.decode(second_tokens, skip_special_tokens=True)
            second_eval = _native_eval(
                parser, second_output, item, item["Attacker Tools"][1]
            )

        final_eval = first["eval"] if case["subgroup"] == "direct_harm" else (
            second_eval["eval"] if first["eval"] == "succ" and second_eval else first["eval"]
        )
        native_valid = final_eval != "invalid"
        native_success = (
            first["eval"] == "succ"
            if case["subgroup"] == "direct_harm"
            else bool(second_eval and second_eval["eval"] == "succ")
        )
        save_record(
            output_path,
            {
                **identity,
                **{key: value for key, value in case.items() if key != "item"},
                "case_hash": case_hash,
                "prompt_hash": hash_messages(messages),
                **scorer.score(residual, scorer.dictionary),
                "injection_exposed": True,
                "generated_response": output,
                "second_generated_response": second_output,
                "native_valid": native_valid,
                "native_utility": None,
                "native_attack_success": native_success,
                "native_step1_result": first["eval"],
                "native_step2_result": second_eval["eval"] if second_eval else None,
            },
        )
