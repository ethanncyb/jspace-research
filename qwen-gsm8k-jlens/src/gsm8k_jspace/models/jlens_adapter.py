"""J-Lens adapter used by capture and intervention."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import jlens

from gsm8k_jspace.config import AppConfig


class MissingJLensError(RuntimeError):
    """Fitted J-Lens could not be loaded or does not match the model."""


class JLensAdapter:
    def __init__(self, lens, *, source_desc: str, placeholder: bool = False) -> None:
        self._lens = lens
        self._pinv: dict[int, torch.Tensor] = {}
        self.placeholder = placeholder
        self._metadata = {
            "source": source_desc,
            "placeholder": placeholder,
            "status": "identity-placeholder" if placeholder else "fitted",
            "supported_layers": list(lens.source_layers),
            "n_layers_fitted": len(lens.source_layers),
            "d_model": lens.d_model,
            "n_prompts": lens.n_prompts,
        }

    def project_to_jspace(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self._lens.transport(hidden_states.float(), layer_idx)

    def project_from_jspace(
        self,
        jspace_states: torch.Tensor,
        layer_idx: int,
        *,
        compute_device: str = "cpu",
    ) -> torch.Tensor:
        if layer_idx not in self._pinv:
            J = self._lens.jacobians[layer_idx].float()
            device = torch.device(compute_device)
            self._pinv[layer_idx] = torch.linalg.pinv(J.to(device)).cpu()
        pinv_T = self._pinv[layer_idx].T.to(jspace_states.device)
        return jspace_states.float() @ pinv_T

    def get_supported_layers(self) -> list[int]:
        return list(self._lens.source_layers)

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def top_jspace_tokens(
        self, hidden_state: torch.Tensor, layer_idx: int, lens_model, k: int
    ) -> list[dict[str, Any]]:
        z = self.project_to_jspace(hidden_state, layer_idx)
        return topk_token_rows(lens_model.unembed(z), lens_model.tokenizer, k)

    def top_logit_tokens(self, hidden_state: torch.Tensor, lens_model, k: int) -> list[dict[str, Any]]:
        return topk_token_rows(lens_model.unembed(hidden_state), lens_model.tokenizer, k)


def topk_token_rows(logits: torch.Tensor, tokenizer, k: int) -> list[dict[str, Any]]:
    vec = logits.float().reshape(-1)
    count = min(int(k), int(vec.numel()))
    values, indices = vec.topk(count)
    rows = []
    decode = getattr(tokenizer, "decode", None)
    for token_id, logit in zip(indices.tolist(), values.tolist()):
        text = decode([int(token_id)]) if decode is not None else str(token_id)
        rows.append({"token_id": int(token_id), "text": text, "logit": float(logit)})
    return rows


def _identity_lens(n_layers: int, d_model: int):
    eye = torch.eye(d_model)
    return jlens.JacobianLens(
        {layer: eye.clone() for layer in range(n_layers)},
        n_prompts=0,
        d_model=d_model,
    )


def _as_lens_model(hf_model, tokenizer):
    if all(
        hasattr(hf_model, name)
        for name in ("layers", "n_layers", "unembed", "d_model")
    ):
        if getattr(hf_model, "tokenizer", None) is None:
            hf_model.tokenizer = tokenizer
        return hf_model
    return jlens.from_hf(hf_model, tokenizer)


def load_jlens(cfg: AppConfig, hf_model, tokenizer) -> tuple[JLensAdapter, Any]:
    lens_model = _as_lens_model(hf_model, tokenizer)
    lens_cfg = cfg.jlens
    if lens_cfg.source == "identity":
        print(
            "[load_jlens] WARNING: identity placeholder — NOT a fitted J-Lens"
        )
        lens = _identity_lens(lens_model.n_layers, lens_model.d_model)
        return (
            JLensAdapter(lens, source_desc="identity placeholder", placeholder=True),
            lens_model,
        )
    try:
        if lens_cfg.source == "hub":
            lens = jlens.JacobianLens.from_pretrained(
                lens_cfg.repo,
                filename=lens_cfg.file,
                revision=lens_cfg.revision,
            )
            desc = f"{lens_cfg.repo}@{lens_cfg.revision}:{lens_cfg.file}"
        elif lens_cfg.source == "local":
            lens = jlens.JacobianLens.from_pretrained(str(Path(lens_cfg.local_path)))
            desc = str(lens_cfg.local_path)
        else:
            raise MissingJLensError(f"unknown jlens source {lens_cfg.source!r}")
    except MissingJLensError:
        raise
    except Exception as exc:
        raise MissingJLensError(
            "the fitted J-Lens checkpoint could not be loaded "
            f"(config={lens_cfg}). Original error: {exc}"
        ) from exc

    if lens.d_model != lens_model.d_model:
        raise MissingJLensError(
            f"lens d_model={lens.d_model} != model d_model={lens_model.d_model}"
        )
    adapter = JLensAdapter(lens, source_desc=desc)
    print(f"[load_jlens] {lens} — {desc}")
    return adapter, lens_model
