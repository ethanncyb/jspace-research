from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

from ..phase1.data import hash_messages, render_ids
from ..runtime import append_jsonl
from .common import content_hash, require_generation_context, save_record

HARNESS_VERSION = 5

_GEMMA_CALL_STARTS = (
    re.compile(r"call:([A-Za-z_][A-Za-z0-9_]*)\s*\{"),
    re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)\s*[=>]?\s*\{"),
)
_GEMMA_STRING_ESCAPE = '<|"|>'
_THINKING_OPEN = "<|channel>"
_THINKING_CLOSE = "<channel|>"
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)
_JSON_ESCAPE = re.compile(r"\\(.)", re.DOTALL)
_BARE_KEY = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:")
_NATIVE_CALL_OPEN = re.compile(r"<function\s*=\s*[^>]+>")
_NATIVE_CALL_CLOSE = "</function>"


def _repair_escapes(string: str) -> str:
    """Keep a backslash literal when it does not start a JSON escape, as in ``\\ come``."""

    return _JSON_ESCAPE.sub(
        lambda match: match.group() if match.group(1) in '"\\/bfnrtu' else "\\" + match.group(),
        string,
    )


def _thinking_token_ids(tokenizer: Any) -> tuple[int, int] | None:
    ids = []
    for token in (_THINKING_OPEN, _THINKING_CLOSE):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or tokenizer.convert_ids_to_tokens(token_id) != token:
            return None
        ids.append(token_id)
    return ids[0], ids[1]


def _without_thinking(token_ids: list[int], thinking_ids: tuple[int, int] | None) -> list[int]:
    """Drop Gemma thinking-channel spans.

    Decoding with ``skip_special_tokens`` removes only the channel markers, so the
    channel name ``thought`` would otherwise leak into the assistant history.
    """

    if thinking_ids is None:
        return token_ids
    open_id, close_id = thinking_ids
    kept: list[int] = []
    inside = False
    for token_id in token_ids:
        if token_id == open_id:
            inside = True
        elif token_id == close_id:
            inside = False
        elif not inside:
            kept.append(token_id)
    return kept


def _gemma_arguments(text: str, start: int) -> dict[str, Any] | None:
    """Parse the one Gemma argument object that opens at ``text[start]``."""

    pieces: list[str] = []
    depth = 0
    expect_key = False
    index = start
    while index < len(text):
        if text.startswith(_GEMMA_STRING_ESCAPE, index):
            end = text.find(_GEMMA_STRING_ESCAPE, index + len(_GEMMA_STRING_ESCAPE))
            if end == -1:
                return None
            pieces.append(json.dumps(text[index + len(_GEMMA_STRING_ESCAPE) : end]))
            index = end + len(_GEMMA_STRING_ESCAPE)
            expect_key = False
            continue
        if text[index] == '"':
            string = _JSON_STRING.match(text, index)
            if string is None:
                return None
            pieces.append(_repair_escapes(string.group()))
            index = string.end()
            expect_key = False
            continue
        if expect_key and (key := _BARE_KEY.match(text, index)):
            pieces.append(json.dumps(key.group(1)) + ":")
            index = key.end()
            expect_key = False
            continue
        char = text[index]
        pieces.append(char)
        index += 1
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        if char in "{,":
            expect_key = True
        elif not char.isspace():
            expect_key = False
        if depth == 0:
            try:
                value = json.loads("".join(pieces), strict=False)
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def _first_gemma_call(text: str) -> tuple[str, dict[str, Any]] | None:
    matches = [match for pattern in _GEMMA_CALL_STARTS if (match := pattern.search(text))]
    if not matches:
        return None
    match = min(matches, key=lambda item: item.start())
    arguments = _gemma_arguments(text, match.end() - 1)
    return None if arguments is None else (match.group(1), arguments)


def _native_call_parses(completion: str, native: re.Match[str]) -> bool:
    """Mirror AgentDojo's ``_parse_model_output`` so its accepted calls stay untouched."""

    end = completion.find(_NATIVE_CALL_CLOSE, native.end())
    raw_json = completion[native.end() : end if end != -1 else len(completion)].strip()
    try:
        return isinstance(json.loads(raw_json), dict)
    except json.JSONDecodeError:
        return False


def _normalize_gemma_tool_call(completion: str, raw_completion: str | None = None) -> str:
    """Translate Gemma's first tool call into AgentDojo's native form.

    Gemma 4 wraps string arguments in the special token ``<|"|>``, which decoding with
    ``skip_special_tokens`` removes, so the call is parsed from ``raw_completion`` first.
    """

    call = None
    for source in (raw_completion, completion):
        if source is not None and (call := _first_gemma_call(source)) is not None:
            break
    if call is None:
        return completion
    name, arguments = call
    location = re.search(rf"(?:call:|<function=){re.escape(name)}\b", completion)
    native = _NATIVE_CALL_OPEN.search(completion)
    if native is not None and (
        location is None
        or native.start() < location.start()
        or (native.start() == location.start() and _native_call_parses(completion, native))
    ):
        return completion
    prefix = completion[: location.start()] if location is not None else ""
    return (
        f"{prefix}<function={name}>"
        f"{json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))}"
        "</function>"
    )


def _through_first_tool_call(completion: str) -> str:
    """Drop text generated after the first call; AgentDojo executes only that call."""

    match = _NATIVE_CALL_OPEN.search(completion)
    if match is None:
        return completion
    end = completion.find(_NATIVE_CALL_CLOSE, match.end())
    if end == -1:
        return completion
    return completion[: end + len(_NATIVE_CALL_CLOSE)]


def _string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_values(item)


def _decoded_strings(tool_text: str) -> list[str]:
    """Return the tool text plus string values from each format AgentDojo tools emit.

    Tools return either YAML or a Python ``repr``. YAML reads a single-quoted
    repr string without processing its ``\\n`` escapes, so both decoders are needed.
    """

    values = [tool_text]
    for decode in (yaml.safe_load, ast.literal_eval):
        try:
            decoded = decode(tool_text)
        except (yaml.YAMLError, ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        values.extend(_string_values(decoded))
    return values


def _contains_injection(tool_texts: list[str], injected_texts: list[str]) -> bool:
    normalized_injections = [" ".join(text.split()) for text in injected_texts if text.strip()]
    for tool_text in tool_texts:
        if any(
            injection in " ".join(value.split())
            for value in _decoded_strings(tool_text)
            for injection in normalized_injections
        ):
            return True
    return False


def _json_ready(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            default=lambda item: (
                item.model_dump(mode="json") if hasattr(item, "model_dump") else str(item)
            ),
        )
    )


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
                value = get_text_content_as_str(content) if content is not None else "None"
                text = json.dumps({"result": value if value != "None" else "Success"})
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

    thinking_ids = _thinking_token_ids(model.tokenizer)
    # The chat template already opens and closes the empty thinking channel; greedy decoding
    # otherwise reopens it until the token budget runs out.
    suppressed = () if thinking_ids is None else (thinking_ids[0],)

    class LocalModelElement(BasePipelineElement):
        name = "local"

        def __init__(self) -> None:
            self.injection_exposed = False
            self.first_exposed_step: int | None = None
            self.capture: dict[str, Any] | None = None
            self.captured_completion: str | None = None
            self.last_completion = ""
            self.steps: list[dict[str, Any]] = []
            self.messages: list[Any] = []

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
            if exposed and self.first_exposed_step is None:
                self.first_exposed_step = len(self.steps)
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
                    suppress_token_ids=suppressed,
                )
                self.capture = {
                    **scorer.score(residual, scorer.dictionary),
                    "prompt_hash": hash_messages(chat),
                }
            else:
                tokens = model.generate_from_prompt(
                    input_ids, max_new_tokens=max_new_tokens, suppress_token_ids=suppressed
                )
            visible = _without_thinking(tokens.tolist(), thinking_ids)
            self.last_completion = model.tokenizer.decode(visible, skip_special_tokens=True)
            if eligible:
                self.captured_completion = self.last_completion
            raw_completion = model.tokenizer.decode(tokens, skip_special_tokens=False)
            parsed_text = _through_first_tool_call(
                _normalize_gemma_tool_call(
                    self.last_completion,
                    model.tokenizer.decode(visible, skip_special_tokens=False),
                )
            )
            output = _parse_model_output(parsed_text)
            self.steps.append(
                {
                    "step": len(self.steps),
                    "prompt_tokens": int(input_ids.shape[-1]),
                    "generated_tokens": int(tokens.shape[-1]),
                    "injection_exposed": exposed,
                    "captured": eligible,
                    "raw_completion": raw_completion,
                    "completion": self.last_completion,
                    "parsed_text": parsed_text,
                    "parsed_tool_calls": _json_ready(output["tool_calls"] or []),
                }
            )
            self.messages = [*messages, output]
            return query, runtime, env, self.messages, extra_args

    return LocalModelElement()


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
    trajectory_path = config.output_dir / "agentdojo_trajectories.jsonl"
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
            "harness_version": HARNESS_VERSION,
            "user_prompt": str(user_task.PROMPT),
            "injection_goal": (None if injection_task is None else str(injection_task.GOAL)),
            "injections": injections,
        }
        case_hash = content_hash(case_identity)
        if case_id in completed:
            if completed[case_id].get("case_hash") != case_hash:
                raise RuntimeError(
                    f"Cached AgentDojo case identity changed for {case_id}; delete "
                    f"{output_path} to regenerate with the current harness"
                )
            continue
        utility, attack_success = suite.run_task_with_pipeline(
            pipeline, user_task, injection_task, injections
        )
        append_jsonl(
            trajectory_path,
            {
                "case_id": case_id,
                "case_hash": case_hash,
                "harness_version": HARNESS_VERSION,
                "injections": injections,
                "first_exposed_step": tracker.first_exposed_step,
                "steps": tracker.steps,
                "messages": _json_ready(tracker.messages),
            },
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
