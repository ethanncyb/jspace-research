from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from ..runtime import append_jsonl, read_resumable_jsonl, validate_identity_fields

# Part of every agent-benchmark case identity, so records generated before the
# special-token decoding fix cannot be resumed into a run that uses it.
AGENT_DECODING_VERSION = "gemma-special-tokens-v2"

_THINKING_BLOCK = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)
_UNCLOSED_CHANNEL_LABEL = re.compile(r"<\|channel>[A-Za-z_]*\n?")


def decode_with_markup(tokenizer: Any, tokens: Any) -> str:
    """Decode a completion keeping Gemma's control tokens, minus thinking channels.

    Gemma writes string arguments of native tool calls between `<|"|>` special
    tokens, and opens a thinking channel with `<|channel>thought`. Decoding with
    `skip_special_tokens=True` deletes the string delimiters, which makes tool
    arguments unparseable, and leaves a bare "thought" label in the response.
    A closed channel is reasoning, not response, so it is removed whole; an
    unclosed one loses only its label, because its text is all the model wrote.
    """

    text: str = tokenizer.decode(tokens, skip_special_tokens=False)
    text = _THINKING_BLOCK.sub("", text)
    return _UNCLOSED_CHANNEL_LABEL.sub("", text)


_SPECIAL_TOKEN_PATTERNS: dict[int, tuple[Any, re.Pattern[str] | None]] = {}


def _special_token_pattern(tokenizer: Any) -> re.Pattern[str] | None:
    cached: tuple[Any, re.Pattern[str] | None] | None = _SPECIAL_TOKEN_PATTERNS.get(id(tokenizer))
    if cached is not None and cached[0] is tokenizer:
        return cached[1]
    pattern: re.Pattern[str] | None = _build_special_token_pattern(tokenizer)
    _SPECIAL_TOKEN_PATTERNS[id(tokenizer)] = (tokenizer, pattern)
    return pattern


def _build_special_token_pattern(tokenizer: Any) -> re.Pattern[str] | None:
    added: dict[int, Any] = getattr(tokenizer, "added_tokens_decoder", None) or {}
    strings: set[str] = {
        str(token.content) for token in added.values() if getattr(token, "special", False)
    }
    strings.update(str(token) for token in getattr(tokenizer, "all_special_tokens", []) or [])
    strings.discard("")
    if not strings:
        return None
    ordered: list[str] = sorted(strings, key=len, reverse=True)
    return re.compile("|".join(re.escape(value) for value in ordered))


def remove_special_tokens(tokenizer: Any, text: str) -> str:
    pattern: re.Pattern[str] | None = _special_token_pattern(tokenizer)
    return text if pattern is None else pattern.sub("", text)


def decode_completion(tokenizer: Any, tokens: Any) -> str:
    """Decode a completion as plain response text, without thinking or control tokens."""

    return remove_special_tokens(tokenizer, decode_with_markup(tokenizer, tokens))


def verify_checkout(root: Path, expected_revision: str, name: str) -> None:
    try:
        actual = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot verify the {name} checkout at {root}") from exc
    if actual != expected_revision:
        raise RuntimeError(
            f"{name} revision mismatch: expected {expected_revision}, found {actual}"
        )


def content_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def require_generation_context(
    prompt_tokens: int, context_length: int, max_new_tokens: int, benchmark: str
) -> None:
    if prompt_tokens + max_new_tokens > context_length:
        raise RuntimeError(
            f"{benchmark} prompt and generation exceed the pinned model context window: "
            f"{prompt_tokens} + {max_new_tokens} > {context_length} tokens"
        )


def completed_records(path: Path, identity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = read_resumable_jsonl(path)
    completed: dict[str, dict[str, Any]] = {}
    for row in rows:
        validate_identity_fields(path, row, identity)
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in completed:
            raise RuntimeError(f"Duplicate or invalid case ID in {path}")
        if not isinstance(row.get("case_hash"), str) or not isinstance(
            row.get("generated_response"), str
        ):
            raise RuntimeError(f"Incomplete Phase 4 record in {path}")
        detector_values = [
            row.get("mean_score"),
            row.get("mean_prediction"),
            row.get("logistic_score"),
            row.get("logistic_prediction"),
        ]
        missing_detector = all(value is None for value in detector_values)
        if any(value is None for value in detector_values):
            if not missing_detector:
                raise RuntimeError(f"Partially missing detector output in {path}")
        elif not (
            all(math.isfinite(float(value)) for value in detector_values[::2])
            and all(isinstance(value, bool) for value in detector_values[1::2])
        ):
            raise RuntimeError(f"Invalid detector output in {path}")
        benchmark = row.get("benchmark")
        condition = row.get("condition")
        if benchmark not in {"bipia", "agentdojo", "injecagent"} or condition not in {
            "attack",
            "control",
        }:
            raise RuntimeError(f"Invalid benchmark condition in {path}")
        if benchmark != "agentdojo" and missing_detector:
            raise RuntimeError(f"Missing detector output in {path}")
        if benchmark == "bipia" and not isinstance(row.get("context_id"), str):
            raise RuntimeError(f"Missing BIPIA context identity in {path}")
        if benchmark == "agentdojo" and condition == "attack":
            exposed = row.get("injection_exposed")
            if not isinstance(exposed, bool) or exposed == missing_detector:
                raise RuntimeError(f"Inconsistent AgentDojo exposure state in {path}")
        completed[case_id] = row
    return completed


def save_record(path: Path, record: dict[str, Any]) -> None:
    append_jsonl(path, record)
