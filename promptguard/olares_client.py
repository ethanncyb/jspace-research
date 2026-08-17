"""Resilient Olares inference client with lazy local fallbacks.

Every public inference function catches network and backend failures per call.
An unavailable Olares box therefore degrades to local Hugging Face inference
instead of aborting a long evolution or benchmark run.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
import requests

from promptguard.config import OlaresConfig

LOGGER = logging.getLogger(__name__)

OLARES_BASE_URL = os.environ.get(
    "OLARES_BASE_URL", "https://a5be22681.dav50505.olares.com"
)
OLARES_API_KEY = os.environ.get("OLARES_API_KEY", "ollama")

OLARES_EMBED_MODEL = "nomic-embed-text"
OLARES_FAST_MODEL = "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:Q8_0"
OLARES_REASON_MODEL = "glm-fixed"
OLARES_CODE_MODEL = "qwen3-coder:30b"

FALLBACK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

USE_OLARES = False
_ACTIVE_CHAT_API: str | None = None
_CONFIG = OlaresConfig(base_url=OLARES_BASE_URL)
_LOCAL_GENERATOR: Any = None
_LOCAL_EMBEDDER: Any = None
_LOCK = threading.Lock()


def configure(config: OlaresConfig) -> None:
    """Apply YAML configuration while allowing environment overrides."""

    global _CONFIG, OLARES_BASE_URL, OLARES_API_KEY
    env_base = os.environ.get("OLARES_BASE_URL")
    _CONFIG = config
    OLARES_BASE_URL = (env_base or config.base_url).rstrip("/")
    OLARES_API_KEY = os.environ.get(config.api_key_env, config.default_api_key)


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {OLARES_API_KEY}",
        "Content-Type": "application/json",
    }


def _openai_generate(
    prompt: str,
    *,
    model: str,
    system: str | None,
    max_tokens: int,
    timeout: float,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = requests.post(
        f"{OLARES_BASE_URL}/v1/chat/completions",
        headers=_headers(),
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    if not content:
        LOGGER.warning(
            "Olares returned empty content: finish_reason=%r message=%r usage=%r",
            choice.get("finish_reason"),
            message,
            data.get("usage"),
        )
        raise ValueError("Olares returned an empty completion")
    return str(content)


def check_olares_reachable() -> bool:
    """Probe the confirmed-working OpenAI-compatible chat endpoint."""

    global USE_OLARES, _ACTIVE_CHAT_API
    USE_OLARES = False
    _ACTIVE_CHAT_API = None
    if _CONFIG.force_local_fallback:
        LOGGER.info("Olares disabled by force_local_fallback")
        return False
    try:
        health_text = _openai_generate(
            "Reply with OK.",
            model=_CONFIG.fast_model,
            system=None,
            max_tokens=256,
            timeout=max(_CONFIG.timeout_seconds, 10.0),
        )
        if "OK" not in health_text.upper():
            raise ValueError(f"unexpected health response: {health_text!r}")
    except Exception as exc:  # network/protocol errors are expected here
        LOGGER.debug("Olares OpenAI chat health probe failed: %s", exc)
    else:
        USE_OLARES = True
        _ACTIVE_CHAT_API = "openai"
        LOGGER.info("Olares reachable through OpenAI chat API")
        return True
    LOGGER.warning("Olares unavailable; using local Hugging Face fallbacks")
    return False


def _get_local_generator():
    global _LOCAL_GENERATOR
    if _LOCAL_GENERATOR is None:
        with _LOCK:
            if _LOCAL_GENERATOR is None:
                from transformers import pipeline

                _LOCAL_GENERATOR = pipeline(
                    "text-generation",
                    model=_CONFIG.fallback_llm_model,
                    device_map="auto",
                    dtype="auto",
                )
    return _LOCAL_GENERATOR


def _local_generate(prompt: str, *, system: str | None, max_tokens: int) -> str:
    full_prompt = f"System: {system}\n\nUser: {prompt}" if system else prompt
    output = _get_local_generator()(
        full_prompt,
        max_new_tokens=max_tokens,
        do_sample=False,
        return_full_text=False,
    )
    if not output:
        return ""
    generated = output[0].get("generated_text", "")
    if isinstance(generated, list):
        generated = generated[-1].get("content", "")
    return str(generated)


def generate(
    prompt: str,
    model: str = OLARES_FAST_MODEL,
    system: str | None = None,
    max_tokens: int = 512,
) -> str:
    """Generate with Olares, falling back locally on every failed call."""

    if USE_OLARES and not _CONFIG.force_local_fallback:
        try:
            return _openai_generate(
                prompt,
                model=model,
                system=system,
                max_tokens=max_tokens,
                timeout=_CONFIG.timeout_seconds,
            )
        except Exception as exc:
            LOGGER.warning("Olares generation call failed: %s", exc)
    try:
        return _local_generate(prompt, system=system, max_tokens=max_tokens)
    except Exception as exc:
        LOGGER.error("local generation fallback failed; returning empty text: %s", exc)
        return ""


def _native_embed(text: str, model: str) -> list[float]:
    response = requests.post(
        f"{OLARES_BASE_URL}/api/embed",
        headers=_headers(),
        json={"model": model, "input": text},
        timeout=_CONFIG.timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    values = data.get("embeddings", data.get("embedding"))
    if values and isinstance(values[0], list):
        values = values[0]
    return [float(value) for value in values]


def _openai_embed(text: str, model: str) -> list[float]:
    response = requests.post(
        f"{OLARES_BASE_URL}/v1/embeddings",
        headers=_headers(),
        json={"model": model, "input": text},
        timeout=_CONFIG.timeout_seconds,
    )
    response.raise_for_status()
    return [float(value) for value in response.json()["data"][0]["embedding"]]


def _get_local_embedder():
    global _LOCAL_EMBEDDER
    if _LOCAL_EMBEDDER is None:
        with _LOCK:
            if _LOCAL_EMBEDDER is None:
                from sentence_transformers import SentenceTransformer

                _LOCAL_EMBEDDER = SentenceTransformer(_CONFIG.fallback_embed_model)
    return _LOCAL_EMBEDDER


def _hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    """Last-resort deterministic vector so dedup never aborts a run."""

    digest = hashlib.sha256(text.encode()).digest()
    raw = (digest * ((dimensions // len(digest)) + 1))[:dimensions]
    vector = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 127.5
    vector /= np.linalg.norm(vector).clip(min=1e-12)
    return vector.tolist()


def embed(text: str, model: str = OLARES_EMBED_MODEL) -> list[float]:
    """Embed text through Olares, sentence-transformers, or a stable hash."""

    if USE_OLARES and not _CONFIG.force_local_fallback:
        for caller in (_native_embed, _openai_embed):
            try:
                return caller(text, model)
            except Exception as exc:
                LOGGER.warning("Olares embedding call failed: %s", exc)
    try:
        values = _get_local_embedder().encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return [float(value) for value in values]
    except Exception as exc:
        LOGGER.error("local embedding fallback failed; using stable hash: %s", exc)
        return _hash_embedding(text)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def backend_name() -> str:
    return f"olares:{_ACTIVE_CHAT_API}" if USE_OLARES else "local_fallback"
