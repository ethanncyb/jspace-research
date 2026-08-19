"""Versioned GSM8K numeric answer parser (`gsm8k_numeric_v1`)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+)?"
)
CURRENCY_RE = re.compile(r"[$£€¥]")
TRAILING_PUNCT_RE = re.compile(r"[.,;:]+$")


@dataclass(frozen=True)
class ParsedAnswer:
    raw: str | None
    normalized: str | None
    value: Decimal | None
    method: str
    succeeded: bool
    error: str | None = None


def _strip_noise(text: str) -> str:
    text = CURRENCY_RE.sub("", text)
    text = text.replace("%", "")
    return text.strip()


def _canonical(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"-0", "+0"}:
        return "0"
    return normalized


def parse_number_token(token: str) -> Decimal | None:
    cleaned = _strip_noise(token)
    cleaned = TRAILING_PUNCT_RE.sub("", cleaned.strip())
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace(" ", "")
    if not cleaned:
        return None
    try:
        if "/" in cleaned:
            left, right = cleaned.split("/", 1)
            denom = Decimal(right)
            if denom == 0:
                return None
            return Decimal(left) / denom
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def _from_marker(text: str, marker: str) -> tuple[str | None, str]:
    if marker not in text:
        return None, "missing_marker"
    after = text.rsplit(marker, 1)[-1]
    match = NUMBER_RE.search(after)
    if not match:
        return None, "marker_without_number"
    return match.group(0), "answer_marker"


def _last_number(text: str) -> tuple[str | None, str]:
    matches = list(NUMBER_RE.finditer(text))
    if not matches:
        return None, "no_number"
    return matches[-1].group(0), "last_number"


def parse_gsm8k_answer(
    text: str,
    *,
    marker: str = "####",
    prefer_answer_marker: bool = True,
    allow_last_number_fallback: bool = True,
) -> ParsedAnswer:
    if text is None:
        return ParsedAnswer(None, None, None, "empty", False, "empty_text")
    source = str(text)
    raw = None
    method = "empty"
    if prefer_answer_marker:
        raw, method = _from_marker(source, marker)
    if raw is None and allow_last_number_fallback:
        raw, method = _last_number(source)
    if raw is None:
        return ParsedAnswer(None, None, None, method, False, method)
    value = parse_number_token(raw)
    if value is None:
        return ParsedAnswer(raw, None, None, method, False, "unparseable_number")
    return ParsedAnswer(raw, _canonical(value), value, method, True)


def answers_match(
    predicted: ParsedAnswer,
    gold: ParsedAnswer,
    *,
    tolerance: float = 0.0,
) -> bool:
    if not predicted.succeeded or not gold.succeeded:
        return False
    assert predicted.value is not None and gold.value is not None
    if tolerance <= 0:
        return predicted.normalized == gold.normalized
    return abs(predicted.value - gold.value) <= Decimal(str(tolerance))
