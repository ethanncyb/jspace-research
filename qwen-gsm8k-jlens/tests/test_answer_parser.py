from __future__ import annotations

from gsm8k_jspace.evaluation.answer_parser import answers_match, parse_gsm8k_answer, parse_number_token


def test_marker_preferred():
    parsed = parse_gsm8k_answer("scratch 12 then #### 18")
    assert parsed.succeeded
    assert parsed.normalized == "18"
    assert parsed.method == "answer_marker"


def test_last_number_fallback():
    parsed = parse_gsm8k_answer("I think the value is 7", prefer_answer_marker=True)
    assert parsed.normalized == "7"
    assert parsed.method == "last_number"


def test_commas_currency_decimals_negatives():
    assert parse_number_token("$1,234.50") == parse_number_token("1234.50")
    parsed = parse_gsm8k_answer("#### -3")
    assert parsed.normalized == "-3"


def test_fractions_and_tolerance():
    gold = parse_gsm8k_answer("#### 1/2")
    pred = parse_gsm8k_answer("#### 0.5")
    assert answers_match(pred, gold, tolerance=0.0)
    almost = parse_gsm8k_answer("#### 0.5001")
    assert not answers_match(almost, gold, tolerance=0.0)
    assert answers_match(almost, gold, tolerance=0.001)


def test_empty_and_malformed():
    empty = parse_gsm8k_answer("")
    assert not empty.succeeded
    bad = parse_gsm8k_answer("#### not-a-number")
    assert not bad.succeeded


def test_scientific_notation_allowed():
    parsed = parse_gsm8k_answer("#### 1e3")
    assert parsed.succeeded
    assert parsed.normalized == "1000"
