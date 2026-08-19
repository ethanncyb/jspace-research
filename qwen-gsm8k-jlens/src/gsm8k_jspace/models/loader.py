"""Model and tokenizer loading driven by config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import transformers

from gsm8k_jspace.config import AppConfig
from gsm8k_jspace.platform.capabilities import inspect_backend
from gsm8k_jspace.platform.device import resolve_torch_device, tensor_device
from gsm8k_jspace.platform.memory import memory_preflight


@dataclass
class ModelBundle:
    hf_model: Any
    tokenizer: Any
    backend: str
    dtype: torch.dtype
    device: torch.device

    @property
    def input_device(self) -> torch.device:
        try:
            return self.hf_model.get_input_embeddings().weight.device
        except Exception:
            return tensor_device(self.hf_model)


def load_hf_model(cfg: AppConfig) -> ModelBundle:
    info = inspect_backend(cfg.runtime.backend, cfg.model.dtype)
    device = resolve_torch_device(cfg.runtime.device, info.name)
    memory_preflight(cfg)
    kwargs: dict[str, Any] = {
        "dtype": info.dtype,
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if cfg.model.revision:
        kwargs["revision"] = cfg.model.revision
    if cfg.model.attention_implementation:
        kwargs["attn_implementation"] = cfg.model.attention_implementation
    device_map = cfg.model.device_map
    if info.name == "mps" and device_map == "auto":
        device_map = None
        kwargs["device_map"] = None
    elif device_map:
        kwargs["device_map"] = device_map
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model.name, **kwargs
    )
    if device_map in {None, "mps", "cpu", "cuda"} or info.name == "mps":
        hf_model = hf_model.to(device)
    hf_model.eval()
    tok_kwargs: dict[str, Any] = {}
    if cfg.model.tokenizer_revision:
        tok_kwargs["revision"] = cfg.model.tokenizer_revision
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.name, **tok_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print(
        f"[load_model] {cfg.model.name} backend={info.name} dtype={info.dtype_name} "
        f"device_map={device_map}"
    )
    return ModelBundle(
        hf_model=hf_model,
        tokenizer=tokenizer,
        backend=info.name,
        dtype=info.dtype,
        device=device,
    )


def load_model_bundle(cfg: AppConfig, *, hf_model=None, tokenizer=None) -> ModelBundle:
    if hf_model is not None and tokenizer is not None:
        info = inspect_backend(cfg.runtime.backend, cfg.model.dtype)
        device = tensor_device(hf_model)
        return ModelBundle(
            hf_model=hf_model,
            tokenizer=tokenizer,
            backend=info.name,
            dtype=info.dtype,
            device=device,
        )
    return load_hf_model(cfg)
