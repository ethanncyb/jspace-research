from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from ..phase1.data import hash_messages, render_ids
from .common import content_hash, require_generation_context, save_record

_GEMMA_CALL = re.compile(r"call:([A-Za-z_][A-Za-z0-9_]*)\s*(\{[^\n]*\})")
_GEMMA_FUNCTION_TAG = re.compile(
    r"<function=([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\{[^\n]*\})>\s*</function>"
)
_BARE_ARGUMENT_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")


def _normalize_gemma_tool_call(completion: str) -> str:
    """Translate Gemma's observed tool-call spelling to AgentDojo's native form."""

    match = _GEMMA_FUNCTION_TAG.search(completion) or _GEMMA_CALL.search(completion)
    if match is None:
        return completion
    function_name, raw_arguments = match.groups()
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        quoted_keys = _BARE_ARGUMENT_KEY.sub(r'\1"\2":', raw_arguments)
        try:
            arguments = json.loads(quoted_keys)
        except json.JSONDecodeError:
            return completion
    if not isinstance(arguments, dict):
        return completion
    normalized = (
        f"<function={function_name}>"
        f"{json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))}"
        "</function>"
    )
    return completion[: match.start()] + normalized + completion[match.end() :]


def _string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_values(item)


def _contains_injection(tool_texts: list[str], injected_texts: list[str]) -> bool:
    for tool_text in tool_texts:
        try:
            decoded = yaml.safe_load(tool_text)
        except yaml.YAMLError:
            decoded = tool_text
        if any(
            injection in value
            for value in _string_values(decoded)
            for injection in injected_texts
        ):
            return True
    return False


def _install_checkout(root: Path) -> None:
    source = str((root / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


def _chat_messages(messages: Sequence[Any], runtime: Any) -> list[dict[str, str]]:
    from agentdojo.agent_pipeline.llms.local_llm import _make_system_prompt
    from agentdojo.types import get_text_content_as_str

    converted: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        if role == "system":
            text = _make_system_prompt(
                get_text_content_as_str(content), runtime.functions.values()
            )
            target_role = "system"
        elif role == "tool":
            if message.get("error") is not None:
                text = json.dumps({"error": message["error"]})
            else:
                value = content if content != "None" else "Success"
                text = json.dumps({"result": value})
            target_role = "user"
        else:
            text = get_text_content_as_str(content) if content is not None else ""
            target_role = "assistant" if role == "assistant" else "user"

        if converted and converted[-1]["role"] == target_role and target_role != "system":
            converted[-1]["content"] += "\n\n" + text
        else:
            converted.append({"role": target_role, "content": text})
    return converted


def _make_llm(
    model: Any,
    scorer: Any,
    condition: str,
    injected_texts: list[str],
    *,
    context_length: int,
    max_new_tokens: int,
) -> Any:
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.llms.local_llm import _parse_model_output
    from agentdojo.types import get_text_content_as_str

    class GemmaElement(BasePipelineElement):
        name = "local"

        def __init__(self) -> None:
            self.injection_exposed = False
            self.capture: dict[str, Any] | None = None
            self.captured_completion: str | None = None
            self.last_completion = ""

        def query(
            self,
            query: str,
            runtime: Any,
            env: Any,
            messages: Sequence[Any] = (),
            extra_args: dict | None = None,
        ) -> tuple[Any, Any, Any, Sequence[Any], dict]:
            extra_args = {} if extra_args is None else extra_args
            tool_texts = []
            for message in messages:
                if message["role"] == "tool" and message.get("content") is not None:
                    tool_texts.append(get_text_content_as_str(message["content"]))
            exposed = bool(injected_texts and _contains_injection(tool_texts, injected_texts))
            self.injection_exposed = self.injection_exposed or exposed
            eligible = self.capture is None and (
                (condition == "attack" and exposed)
                or (condition == "control" and bool(tool_texts))
            )
            chat = _chat_messages(messages, runtime)
            input_ids = render_ids(model.tokenizer, chat)
            require_generation_context(
                int(input_ids.shape[-1]), context_length, max_new_tokens, "AgentDojo"
            )
            if eligible:
                tokens, residual = model.generate_with_capture(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    layer=scorer.mean["selected_layer"],
                )
                self.capture = {
                    **scorer.score(residual, scorer.dictionary),
                    "prompt_hash": hash_messages(chat),
                }
            else:
                tokens = model.generate_from_prompt(
                    input_ids, max_new_tokens=max_new_tokens
                )
            self.last_completion = model.tokenizer.decode(tokens, skip_special_tokens=True)
            if eligible:
                self.captured_completion = self.last_completion
            output = _parse_model_output(_normalize_gemma_tool_call(self.last_completion))
            return query, runtime, env, [*messages, output], extra_args

    return GemmaElement()


def _native_cases(suite: Any, smoke: bool) -> list[tuple[str, Any, Any | None]]:
    users = sorted(suite.user_tasks.items(), key=lambda item: str(item[0]))
    injections = sorted(suite.injection_tasks.items(), key=lambda item: str(item[0]))
    attacked = [("attack", user, injection) for user in users for injection in injections]
    if smoke:
        return [("control", user, None) for user in users[:2]] + attacked[:2]
    return [("control", user, None) for user in users] + attacked


def validate_smoke_records(records: list[dict[str, Any]], suites: Sequence[str]) -> None:
    for suite in suites:
        suite_records = [row for row in records if row.get("subgroup") == suite]
        clean_scored = any(
            row.get("condition") == "control" and row.get("mean_score") is not None
            for row in suite_records
        )
        attack_scored = any(
            row.get("condition") == "attack"
            and row.get("injection_exposed") is True
            and row.get("mean_score") is not None
            for row in suite_records
        )
        if not clean_scored or not attack_scored:
            raise RuntimeError(
                f"AgentDojo smoke did not reach eligible clean and exposed attack "
                f"decision points for suite {suite} "
                f"(clean_scored={clean_scored}, exposed_attack_scored={attack_scored})"
            )


def generate(
    config: Any,
    model: Any,
    scorer: Any,
    completed: dict[str, dict[str, Any]],
    identity: dict[str, Any],
) -> None:
    _install_checkout(config.agentdojo_root)
    from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.attacks import load_attack
    from agentdojo.task_suite.load_suites import get_suite

    output_path = config.output_dir / "agentdojo_records.jsonl"
    suite_cases = []
    expected_ids: set[str] = set()
    for suite_name in config.agentdojo_suites:
        suite = get_suite(config.agentdojo_version, suite_name)
        for condition, user_entry, injection_entry in _native_cases(suite, config.smoke):
            user_id, user_task = user_entry
            injection_id = None if injection_entry is None else injection_entry[0]
            injection_task = None if injection_entry is None else injection_entry[1]
            case_id = f"agentdojo:{suite_name}:{condition}:{user_id}:{injection_id or 'none'}"
            expected_ids.add(case_id)
            case_basis = {
                "case_id": case_id,
                "suite": suite_name,
                "condition": condition,
                "user_task_id": str(user_id),
                "injection_task_id": None if injection_id is None else str(injection_id),
            }
            suite_cases.append(
                (suite_name, suite, condition, user_task, injection_task, case_basis)
            )
    unexpected = sorted(set(completed) - expected_ids)
    if unexpected:
        raise RuntimeError(f"Unexpected cached AgentDojo case ID: {unexpected[0]}")

    for suite_name, suite, condition, user_task, injection_task, case_basis in suite_cases:
        case_id = case_basis["case_id"]
        tracker = _make_llm(
            model,
            scorer,
            condition,
            [],
            context_length=model.context_length,
            max_new_tokens=config.max_new_tokens,
        )
        pipeline = AgentPipeline.from_config(
            PipelineConfig(
                llm=tracker,
                model_id=None,
                defense=None,
                tool_delimiter="tool",
                system_message_name=None,
                system_message=None,
                tool_output_format=None,
            )
        )
        injections: dict[str, str] = {}
        if injection_task is not None:
            attack = load_attack(config.agentdojo_attack, suite, pipeline)
            injections = attack.attack(user_task, injection_task)
            tracker = _make_llm(
                model,
                scorer,
                condition,
                list(injections.values()),
                context_length=model.context_length,
                max_new_tokens=config.max_new_tokens,
            )
            pipeline = AgentPipeline.from_config(
                PipelineConfig(
                    llm=tracker,
                    model_id=None,
                    defense=None,
                    tool_delimiter="tool",
                    system_message_name=None,
                    system_message=None,
                    tool_output_format=None,
                )
            )
        case_identity = {
            **case_basis,
            "user_prompt": str(user_task.PROMPT),
            "injection_goal": (None if injection_task is None else str(injection_task.GOAL)),
            "injections": injections,
        }
        case_hash = content_hash(case_identity)
        if case_id in completed:
            if completed[case_id].get("case_hash") != case_hash:
                raise RuntimeError("Cached AgentDojo case identity changed")
            continue
        utility, attack_success = suite.run_task_with_pipeline(
            pipeline, user_task, injection_task, injections
        )
        detector = tracker.capture or {
            "mean_score": None,
            "mean_prediction": None,
            "logistic_score": None,
            "logistic_prediction": None,
            "prompt_hash": None,
        }
        save_record(
            output_path,
            {
                **identity,
                **case_basis,
                "case_hash": case_hash,
                "benchmark": "agentdojo",
                "task": None,
                "subgroup": suite_name,
                **detector,
                "injection_exposed": tracker.injection_exposed,
                "generated_response": tracker.captured_completion or tracker.last_completion,
                "native_valid": None,
                "native_utility": bool(utility),
                "native_attack_success": (
                    bool(attack_success) if injection_task is not None else None
                ),
            },
        )
