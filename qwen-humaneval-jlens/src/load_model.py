"""Model + tokenizer loading for the Phase-1 HumanEval baseline.

Everything is driven by the ``model:`` section of config.yaml. The device map
defaults to ``auto`` (HF accelerate spreads the model across available GPUs /
falls back to CPU offload); dtype defaults to bfloat16 as configured.
"""

from __future__ import annotations

import torch
import transformers

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.float16
        return torch.float32
    return _DTYPES[name]


def load_model(cfg: dict):
    """Load the HF causal LM and tokenizer per ``cfg["model"]``.

    Returns ``(hf_model, tokenizer)``. With ``device_map="auto"`` the model
    placement is handled by accelerate; hooks and capture code must read the
    device from the tensors they observe rather than assuming one device.
    """
    model_cfg = cfg["model"]
    dtype = resolve_dtype(model_cfg.get("dtype", "bfloat16"))

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        dtype=dtype,
        device_map=model_cfg.get("device_map", "auto"),
    )
    hf_model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_cfg["name"])

    print(f"[load_model] {model_cfg['name']} dtype={dtype} "
          f"device_map={model_cfg.get('device_map', 'auto')}")
    return hf_model, tokenizer


def deterministic_gen_config(cfg: dict) -> dict:
    """Decoding settings shared by every sample of the baseline run.

    Greedy (do_sample=False); temperature is recorded for metadata only —
    it has no effect under greedy decoding.
    """
    gen = cfg["generation"]
    return {
        "max_new_tokens": int(gen.get("max_new_tokens", 512)),
        "do_sample": False,
        "temperature": float(gen.get("temperature", 0.0)),
        "seed": int(gen.get("seed", 0)),
    }
