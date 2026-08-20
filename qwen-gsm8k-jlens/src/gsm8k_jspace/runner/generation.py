"""Shared generation path used by every experimental condition."""

from __future__ import annotations

import time
from typing import Any

import torch

from gsm8k_jspace.config import AppConfig
from gsm8k_jspace.types import GenerationResult


class _StopOnStrings:
    def __init__(self, tokenizer, prompt_len: int, stop_strings: list[str]) -> None:
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len
        self._stop = stop_strings

    def __call__(self, input_ids, scores=None, **kwargs) -> bool:
        if not self._stop:
            return False
        text = self._tokenizer.decode(
            input_ids[0, self._prompt_len :], skip_special_tokens=True
        )
        return any(stop in text for stop in self._stop)


def _greedy_tiny(model, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int | None):
    generated = []
    current = input_ids
    finish = "max_new_tokens"
    for _ in range(max_new_tokens):
        hidden = model.forward(current)
        residual = (
            hidden.last_hidden_state if hasattr(hidden, "last_hidden_state") else hidden
        )
        if residual.ndim == 3:
            residual = residual[:, -1]
        logits = model.unembed(residual)
        token_id = int(logits[0].argmax().item())
        generated.append(token_id)
        if eos_id is not None and token_id == eos_id:
            finish = "eos"
            break
        next_tok = torch.tensor([[token_id]], device=current.device, dtype=current.dtype)
        current = torch.cat([current, next_tok], dim=1)
    return generated, finish, current


def _generate_mlx(prompt: str, cfg: AppConfig, bundle, tokenizer) -> GenerationResult:
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(cfg.generation.seed)
    mlx_tokenizer = getattr(tokenizer, "_inner", tokenizer)
    mlx_model = bundle.mlx_model or bundle.hf_model.mlx_model
    if hasattr(tokenizer, "__call__"):
        encoded = tokenizer(prompt, return_tensors="pt")
        prompt_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        prompt_ids = prompt_ids[0].tolist()
    else:
        prompt_ids = list(mlx_tokenizer.encode(prompt))
    prompt_len = len(prompt_ids)
    sampler = make_sampler(temp=0.0 if not cfg.generation.do_sample else 0.7)
    start = time.perf_counter()
    generated_ids: list[int] = []
    finish = "max_new_tokens"
    eos_ids = set()
    raw_eos = getattr(mlx_tokenizer, "eos_token_ids", None) or getattr(
        tokenizer, "eos_token_ids", None
    )
    if raw_eos:
        eos_ids.update(int(x) for x in raw_eos)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        eos_ids.add(int(eos_id))
    for response in stream_generate(
        mlx_model,
        mlx_tokenizer,
        prompt_ids,
        max_tokens=cfg.generation.max_new_tokens,
        sampler=sampler,
    ):
        token_id = int(response.token)
        generated_ids.append(token_id)
        if token_id in eos_ids:
            finish = "eos"
            break
        if cfg.generation.stop_strings:
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if any(stop in text for stop in cfg.generation.stop_strings):
                finish = "stop_string"
                break
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_ids = torch.tensor([prompt_ids + generated_ids], dtype=torch.long)
    return GenerationResult(
        generated_text=text,
        generated_token_ids=[int(x) for x in generated_ids],
        n_prompt_tokens=prompt_len,
        finish_reason=finish,
        elapsed_seconds=elapsed,
        prompt=prompt,
        extra={"full_ids": full_ids, "backend": "mlx"},
    )


def generate_completion(
    *,
    prompt: str,
    cfg: AppConfig,
    bundle,
    tokenizer,
) -> GenerationResult:
    if getattr(bundle, "mlx_model", None) is not None:
        return _generate_mlx(prompt, cfg, bundle, tokenizer)
    model = bundle.hf_model
    torch.manual_seed(cfg.generation.seed)
    if hasattr(tokenizer, "__call__"):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = bundle.input_device
    input_ids = input_ids.to(device)
    prompt_len = int(input_ids.shape[1])
    start = time.perf_counter()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if hasattr(model, "generate"):
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.generation.max_new_tokens,
            "do_sample": cfg.generation.do_sample,
            "pad_token_id": eos_id,
        }
        if cfg.generation.stop_strings:
            from transformers import StoppingCriteriaList

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [_StopOnStrings(tokenizer, prompt_len, cfg.generation.stop_strings)]
            )
        output_ids = model.generate(input_ids, **gen_kwargs)
        generated_ids = output_ids[0, prompt_len:].tolist()
        full_ids = output_ids
        finish = "eos" if generated_ids and generated_ids[-1] == eos_id else "max_new_tokens"
    else:
        generated_ids, finish, full_ids = _greedy_tiny(
            model, input_ids, cfg.generation.max_new_tokens, eos_id
        )
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return GenerationResult(
        generated_text=text,
        generated_token_ids=[int(x) for x in generated_ids],
        n_prompt_tokens=prompt_len,
        finish_reason=finish,
        elapsed_seconds=elapsed,
        prompt=prompt,
        extra={"full_ids": full_ids},
    )
