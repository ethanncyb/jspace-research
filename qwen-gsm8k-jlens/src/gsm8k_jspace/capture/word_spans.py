"""Map generated token IDs to whitespace-delimited word-end positions."""

from __future__ import annotations


def word_end_indices(token_texts: list[str]) -> set[int]:
    ends: set[int] = set()
    if not token_texts:
        return ends
    for index, text in enumerate(token_texts):
        nxt = token_texts[index + 1] if index + 1 < len(token_texts) else ""
        if index == len(token_texts) - 1:
            ends.add(index)
            continue
        if nxt[:1].isspace() or (text.endswith((" ", "\n", "\t"))):
            ends.add(index)
        elif text and not text[-1].isalnum() and nxt[:1].isalnum():
            ends.add(index)
    return ends
