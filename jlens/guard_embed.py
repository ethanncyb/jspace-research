# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Deterministic hashed character n-gram embeddings for the security probe.

No model and no RNG: prompts are NFKC-normalized, hashed into a fixed-width
count vector with blake2b, and L2-normalized, so embeddings are reproducible
across processes and runs.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

_NGRAM_SIZES = (3, 4, 5)
_SEPARATORS_RE = re.compile(r"[^\w]+")


def normalize_prompt(text: str) -> str:
    """NFKC-normalize, lowercase, and collapse whitespace/punctuation runs.

    Every run of non-word characters (whitespace and punctuation) becomes a
    single space, and the result is stripped, so superficial variations of
    the same prompt map to the same normalized form.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return _SEPARATORS_RE.sub(" ", text).strip()


def embed_text(text: str, dim: int = 2048) -> np.ndarray:
    """Embed ``text`` as an L2-normalized hashed character n-gram count vector.

    Character n-grams of sizes 3, 4, and 5 are counted over
    :func:`normalize_prompt(text)`; each n-gram is mapped to a bucket via the
    blake2b digest of its UTF-8 bytes modulo ``dim``. Returns a 1-D
    ``np.float64`` array of length ``dim``; deterministic across processes.
    """
    normalized = normalize_prompt(text)
    vec = np.zeros(dim, dtype=np.float64)
    for n in _NGRAM_SIZES:
        for i in range(len(normalized) - n + 1):
            gram = normalized[i : i + n].encode("utf-8")
            bucket = int.from_bytes(
                hashlib.blake2b(gram, digest_size=8).digest(), "little"
            )
            vec[bucket % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0.0:
        vec /= norm
    return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors; 0.0 when either has zero norm."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
