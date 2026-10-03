from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

from ..phase1.data import hash_messages, render_ids
from .common import (
    AGENT_DECODING_VERSION,
    content_hash,
    decode_with_markup,
    remove_special_tokens,
    require_generation_context,
    save_record,
)

GEMMA_STRING_DELIMITER = '<|"|>'
_GEMMA_CALL_START = re.compile(
    r"(?:<\|tool_call>\s*|(?<![A-Za-z0-9_]))call:([A-Za-z_][A-Za-z0-9_]*)\s*(?=\{)"
)
_GEMMA_CALL_END = "<tool_call|>"
_FUNCTION_TAG_WITH_EQUALS = re.compile(
    r"<function=([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\{[^\n]*\})>\s*</function>"
)


class _ArgumentSyntaxError(ValueError):
    pass


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _parse_bare_scalar(text: str, index: int) -> tuple[Any, int]:
    end: int = index
    while end < len(text) and text[end] not in ",}]":
        end += 1
    raw: str = text[index:end].strip()
    if not raw:
        raise _ArgumentSyntaxError(f"Empty value at offset {index}")
    if raw in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[raw], end
    try:
        number: Any = json.loads(raw)
    except json.JSONDecodeError:
        return raw, end
    return (number, end) if isinstance(number, int | float) else (raw, end)


def _parse_gemma_value(text: str, index: int) -> tuple[Any, int]:
    """Parse one value in Gemma's tool-call argument syntax.

    Strings are wrapped in `<|"|>`, keys are bare, and objects and arrays use
    braces and brackets. JSON-quoted strings and unquoted scalars, which Gemma
    also writes when it imitates the prompt's JSON format, are accepted too.
    """

    index = _skip_space(text, index)
    if text.startswith(GEMMA_STRING_DELIMITER, index):
        start: int = index + len(GEMMA_STRING_DELIMITER)
        end: int = text.find(GEMMA_STRING_DELIMITER, start)
        if end == -1:
            raise _ArgumentSyntaxError("Unterminated Gemma string")
        return text[start:end], end + len(GEMMA_STRING_DELIMITER)
    if index >= len(text):
        raise _ArgumentSyntaxError("Missing value")
    if text[index] == "{":
        return _parse_gemma_object(text, index)
    if text[index] == "[":
        items: list[Any] = []
        index = _skip_space(text, index + 1)
        if index < len(text) and text[index] == "]":
            return items, index + 1
        while True:
            item, index = _parse_gemma_value(text, index)
            items.append(item)
            index = _skip_space(text, index)
            if index < len(text) and text[index] == ",":
                index += 1
            elif index < len(text) and text[index] == "]":
                return items, index + 1
            else:
                raise _ArgumentSyntaxError(f"Unterminated array at offset {index}")
    if text[index] == '"':
        try:
            value, end = json.JSONDecoder().raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise _ArgumentSyntaxError(str(exc)) from exc
        return value, end
    return _parse_bare_scalar(text, index)


def _parse_gemma_key(text: str, index: int) -> tuple[str, int]:
    index = _skip_space(text, index)
    if text.startswith(GEMMA_STRING_DELIMITER, index) or text[index : index + 1] == '"':
        key, index = _parse_gemma_value(text, index)
        if not isinstance(key, str):
            raise _ArgumentSyntaxError("Non-string key")
    else:
        end: int = text.find(":", index)
        if end == -1:
            raise _ArgumentSyntaxError(f"Missing key separator at offset {index}")
        key = text[index:end].strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise _ArgumentSyntaxError(f"Invalid bare key {key!r}")
        index = end
    index = _skip_space(text, index)
    if not text.startswith(":", index):
        raise _ArgumentSyntaxError(f"Missing key separator at offset {index}")
    return key, index + 1


def _parse_gemma_object(text: str, index: int) -> tuple[dict[str, Any], int]:
    if not text.startswith("{", index):
        raise _ArgumentSyntaxError(f"Expected an object at offset {index}")
    values: dict[str, Any] = {}
    index = _skip_space(text, index + 1)
    if index < len(text) and text[index] == "}":
        return values, index + 1
    while True:
        key, index = _parse_gemma_key(text, index)
        values[key], index = _parse_gemma_value(text, index)
        index = _skip_space(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
        elif index < len(text) and text[index] == "}":
            return values, index + 1
        else:
            raise _ArgumentSyntaxError(f"Unterminated object at offset {index}")


def _split_gemma_tool_calls(completion: str) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    """Extract every Gemma `call:name{...}` from a completion decoded with markup.

    Returns the calls in order and the completion text with the calls removed.
    A call whose arguments do not parse is left in the text untouched.
    """

    calls: list[tuple[str, dict[str, Any]]] = []
    pieces: list[str] = []
    cursor: int = 0
    for match in _GEMMA_CALL_START.finditer(completion):
        if match.start() < cursor:
            continue
        try:
            arguments, end = _parse_gemma_object(completion, match.end())
        except _ArgumentSyntaxError:
            continue
        if completion.startswith(_GEMMA_CALL_END, end):
            end += len(_GEMMA_CALL_END)
        calls.append((match.group(1), arguments))
        pieces.append(completion[cursor : match.start()])
        cursor = end
    pieces.append(completion[cursor:])
    return calls, "".join(pieces)


def _render_function_call(name: str, arguments: dict[str, Any]) -> str:
    return (
        f"<function={name}>"
        f"{json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))}"
        "</function>"
    )


def _normalize_function_tag(text: str) -> str:
    """Rewrite the observed `<function=name={...}></function>` variant to the native form."""

    match: re.Match[str] | None = _FUNCTION_TAG_WITH_EQUALS.search(text)
    if match is None:
        return text
    try:
        arguments: Any = json.loads(match.group(2))
    except json.JSONDecodeError:
        return text
    if not isinstance(arguments, dict):
        return text
    rendered: str = _render_function_call(match.group(1), arguments)
    return text[: match.start()] + rendered + text[match.end() :]


def _assistant_message(completion: str, tokenizer: Any) -> Any:
    """Build AgentDojo's assistant message from a completion decoded with markup.

    Gemma's native calls are all executed, as AgentDojo's own provider pipelines
    do for parallel calls. Without native calls, the text goes to AgentDojo's
    prompted-model parser, which reads the first `<function=...>` tag. The
    message text shows each call in the prompt's `<function=...>` form, so the
    model sees its earlier calls in that form on later turns.
    """

    from agentdojo.agent_pipeline.llms.local_llm import _parse_model_output
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.types import ChatAssistantMessage, text_content_block_from_string
    from pydantic import ValidationError

    calls, remainder = _split_gemma_tool_calls(completion)
    text: str = remove_special_tokens(tokenizer, remainder).strip()
    if not calls:
        return _parse_model_output(_normalize_function_tag(text))
    try:
        tool_calls: list[Any] = [
            FunctionCall(function=name, args=arguments) for name, arguments in calls
        ]
    except ValidationError:
        return _parse_model_output(remove_special_tokens(tokenizer, completion).strip())
    rendered: str = "\n".join(_render_function_call(name, arguments) for name, arguments in calls)
    content: str = f"{text}\n{rendered}" if text else rendered
    return ChatAssistantMessage(
        role="assistant",
        content=[text_content_block_from_string(content)],
        tool_calls=tool_calls,
    )


def _string_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _string_values(item)


def _decoded_tool_strings(tool_text: str) -> Iterator[str]:
    """Yield the strings a tool output carries, under each formatting AgentDojo uses.

    Models and lists are dumped as YAML, but `tool_result_to_str` falls back to
    `str()` for dicts, which prints newlines as `\\n` escapes inside a Python
    literal. Matching the raw text alone would miss every multi-line injection
    returned by a dict-valued tool, such as the travel review tools.
    """

    yield tool_text
    try:
        yield from _string_values(yaml.safe_load(tool_text))
    except yaml.YAMLError:
        pass
    try:
        literal: Any = ast.literal_eval(tool_text)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return
    yield from _string_values(literal)


def _contains_injection(tool_texts: list[str], injected_texts: list[str]) -> bool:
    normalized_injections: list[str] = [
        " ".join(text.split()) for text in injected_texts if text.strip()
    ]
    for tool_text in tool_texts:
        if any(
            injection in " ".join(value.split())
            for value in _decoded_tool_strings(tool_text)
            for injection in normalized_injections
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
            output = _assistant_message(
                decode_with_markup(model.tokenizer, tokens), model.tokenizer
            )
            self.last_completion = get_text_content_as_str(output["content"])
            if eligible:
                self.captured_completion = self.last_completion
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

    progress = tqdm(
        total=len(suite_cases), initial=len(completed), desc="Phase 4 AgentDojo generation"
    )
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
            "decoding": AGENT_DECODING_VERSION,
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
        progress.update()
    progress.close()
