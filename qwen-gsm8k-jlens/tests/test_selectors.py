from __future__ import annotations

from gsm8k_jspace.capture.selectors import resolve_layers, should_keep_position
from gsm8k_jspace.capture.word_spans import word_end_indices
from gsm8k_jspace.config import LayerSelector, TokenSelector


def test_late_band():
    layers = resolve_layers(LayerSelector(mode="late"), n_layers=32, fitted_layers=list(range(32)))
    assert layers[0] == 10
    assert layers[-1] == 26


def test_explicit_unknown_layer():
    try:
        resolve_layers(LayerSelector(mode="explicit", values=[99]), 4, [0, 1, 2, 3])
    except ValueError as exc:
        assert "99" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_token_modes():
    prompt = TokenSelector(mode="prompt_last")
    assert should_keep_position(prompt, call_index=0, generated_position=None, is_prefill=True)
    assert not should_keep_position(prompt, call_index=1, generated_position=0, is_prefill=False)
    all_gen = TokenSelector(mode="all_generated")
    assert should_keep_position(all_gen, call_index=1, generated_position=0, is_prefill=False)
    full = TokenSelector(mode="full_sequence")
    assert not should_keep_position(full, call_index=0, generated_position=None, is_prefill=True)
    assert not should_keep_position(full, call_index=1, generated_position=0, is_prefill=False)
    stride = TokenSelector(mode="generated_stride", stride=2)
    assert should_keep_position(stride, call_index=1, generated_position=0, is_prefill=False)
    assert not should_keep_position(stride, call_index=2, generated_position=1, is_prefill=False)


def test_word_end_indices():
    texts = ["Hello", " world", ",", " yes"]
    ends = word_end_indices(texts)
    assert 0 in ends or 1 in ends
    assert max(ends) == len(texts) - 1
