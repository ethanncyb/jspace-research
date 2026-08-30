"""Chat-template wrapping shared by every benchmark."""

from __future__ import annotations

from typing import Any

from gsm8k_jspace.config import AppConfig


def format_model_prompt(
    user_text: str,
    tokenizer,
    cfg: AppConfig,
    *,
    messages: list[dict[str, Any]] | None = None,
) -> str:
    """Optionally wrap the scientific prompt in the tokenizer chat template."""
    if not cfg.prompt.use_chat_template:
        return user_text
    inner = getattr(tokenizer, "_inner", tokenizer)
    apply = getattr(inner, "apply_chat_template", None) or getattr(
        tokenizer, "apply_chat_template", None
    )
    if apply is None:
        return user_text
    payload = messages or [{"role": "user", "content": user_text}]
    rendered = apply(
        payload,
        add_generation_prompt=True,
        tokenize=False,
    )
    if isinstance(rendered, list):
        decode = getattr(inner, "decode", None) or getattr(tokenizer, "decode", None)
        return decode(rendered) if decode is not None else user_text
    return str(rendered)
