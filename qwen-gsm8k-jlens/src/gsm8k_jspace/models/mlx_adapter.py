"""MLX model loading, tokenizer adapter, and hookable decoder layers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from gsm8k_jspace.config import AppConfig


_LAYER_PATHS = (
    ("model", "layers"),
    ("model", "language_model", "layers"),
    ("language_model", "model", "layers"),
    ("language_model", "layers"),
    ("model", "model", "layers"),
    ("layers",),
)

_EMBED_PATHS = (
    ("model", "embed_tokens"),
    ("model", "language_model", "embed_tokens"),
    ("language_model", "model", "embed_tokens"),
    ("language_model", "embed_tokens"),
    ("embed_tokens",),
)

_NORM_PATHS = (
    ("model", "norm"),
    ("model", "language_model", "norm"),
    ("language_model", "model", "norm"),
    ("language_model", "norm"),
    ("model", "final_layernorm"),
    ("norm",),
)


class MissingMLXError(RuntimeError):
    """mlx / mlx-lm is required for the Apple MLX backend."""


def _resolve(obj: Any, path: tuple[str, ...]) -> Any:
    current = obj
    for name in path:
        current = getattr(current, name)
    return current


def _first_path(obj: Any, paths: tuple[tuple[str, ...], ...]):
    last_error: Exception | None = None
    for path in paths:
        try:
            return _resolve(obj, path)
        except AttributeError as exc:
            last_error = exc
    raise AttributeError(f"none of {paths} resolved on {type(obj).__name__}: {last_error}")


def mx_to_torch(array) -> torch.Tensor:
    import mlx.core as mx

    mx.eval(array)
    # NumPy has no bfloat16 buffer format; capture/J-lens math is float32 anyway.
    if getattr(array, "dtype", None) in {mx.bfloat16, mx.float16}:
        array = array.astype(mx.float32)
        mx.eval(array)
    return torch.from_numpy(np.array(array))


def torch_to_mx(tensor: torch.Tensor):
    import mlx.core as mx

    return mx.array(np.ascontiguousarray(tensor.detach().cpu().numpy()))


class _HookHandle:
    def __init__(self, hooks: list, fn) -> None:
        self._hooks = hooks
        self._fn = fn

    def remove(self) -> None:
        if self._fn in self._hooks:
            self._hooks.remove(self._fn)


class MlxHookableBlock:
    """Wrap an MLX decoder block so PyTorch-style forward hooks can fire.

    ``layer()`` uses the class ``__call__``, so instance monkeypatches are
    ignored. The wrapper is spliced into the model's ``layers`` list instead.
    """

    def __init__(self, mlx_layer) -> None:
        object.__setattr__(self, "_layer", mlx_layer)
        object.__setattr__(self, "_hooks", [])

    def __getattr__(self, name: str):
        return getattr(self._layer, name)

    def register_forward_hook(self, fn):
        self._hooks.append(fn)
        return _HookHandle(self._hooks, fn)

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        if not self._hooks:
            return out
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        torch_hidden = mx_to_torch(hidden)
        added_batch = False
        if torch_hidden.ndim == 2:
            torch_hidden = torch_hidden.unsqueeze(0)
            added_batch = True
        result = torch_hidden
        for hook in list(self._hooks):
            maybe = hook(self, args, result)
            if maybe is not None:
                result = maybe[0] if isinstance(maybe, (tuple, list)) else maybe
        if result is torch_hidden:
            return out
        mx_hidden = torch_to_mx(result)
        if added_batch and mx_hidden.ndim == 3 and int(mx_hidden.shape[0]) == 1:
            mx_hidden = mx_hidden[0]
        if isinstance(out, tuple):
            return (mx_hidden, *out[1:])
        if isinstance(out, list):
            return [mx_hidden, *out[1:]]
        return mx_hidden


class TokenizerAdapter:
    """Give mlx-lm tokenizers the HuggingFace-style call used by the runner."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def encode(self, text: str, **kwargs):
        return self._inner.encode(text, **kwargs)

    def __call__(self, text: str, return_tensors: str | None = None, **kwargs):
        ids = list(self._inner.encode(text))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return ids

    def decode(self, ids, skip_special_tokens: bool = True, **kwargs) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(x) for x in ids]
        try:
            return self._inner.decode(ids, skip_special_tokens=skip_special_tokens)
        except TypeError:
            return self._inner.decode(ids)

    @property
    def eos_token_id(self):
        value = getattr(self._inner, "eos_token_id", None)
        if value is not None:
            return value
        ids = getattr(self._inner, "eos_token_ids", None)
        if ids:
            return next(iter(ids))
        return None

    @property
    def pad_token_id(self):
        return getattr(self._inner, "pad_token_id", self.eos_token_id)

    @property
    def eos_token_ids(self):
        ids = getattr(self._inner, "eos_token_ids", None)
        if ids:
            return ids
        eos = self.eos_token_id
        return {eos} if eos is not None else set()


class MlxLensModel:
    """LensModel-shaped wrapper around a loaded mlx-lm / mlx-vlm model."""

    def __init__(self, mlx_model, tokenizer, *, library: str) -> None:
        self._mlx_model = mlx_model
        self.tokenizer = tokenizer
        self.library = library
        raw_layers = _first_path(mlx_model, _LAYER_PATHS)
        self.layers = []
        for index, layer in enumerate(list(raw_layers)):
            if isinstance(layer, MlxHookableBlock):
                wrapped = layer
            else:
                wrapped = MlxHookableBlock(layer)
                try:
                    raw_layers[index] = wrapped
                except (TypeError, AttributeError) as exc:
                    raise RuntimeError(
                        "could not splice MLX decoder wrappers into model.layers"
                    ) from exc
            self.layers.append(wrapped)
        self.n_layers = len(self.layers)
        self._embed = _first_path(mlx_model, _EMBED_PATHS)
        try:
            self._final_norm = _first_path(mlx_model, _NORM_PATHS)
        except AttributeError:
            self._final_norm = None
        self._lm_head = getattr(mlx_model, "lm_head", None)
        self.d_model = _infer_d_model(mlx_model, self._embed)
        if self.n_layers < 1 or self.d_model < 1:
            raise RuntimeError(
                f"MLX lens wrap failed: n_layers={self.n_layers} d_model={self.d_model}"
            )

    @property
    def mlx_model(self):
        return self._mlx_model

    def forward(self, input_ids: torch.Tensor):
        import mlx.core as mx

        ids = input_ids
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        tokens = mx.array(ids)[None]
        return self._mlx_model(tokens)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        import mlx.core as mx

        squeezed = residual.ndim == 1
        array = torch_to_mx(residual.float())
        if array.ndim == 1:
            array = array[None]
        if self._final_norm is not None:
            array = self._final_norm(array)
        if self._lm_head is not None:
            logits = self._lm_head(array)
        else:
            weight = self._embed.weight
            logits = array @ weight.T
        mx.eval(logits)
        out = mx_to_torch(logits).float()
        if squeezed:
            out = out[0]
        return out


def _infer_d_model(model, embed) -> int:
    for obj in (
        model,
        getattr(model, "args", None),
        getattr(model, "config", None),
        getattr(model, "model", None),
        getattr(getattr(model, "args", None), "text_config", None),
    ):
        if obj is None:
            continue
        mapping = obj if isinstance(obj, dict) else None
        if mapping is not None:
            for key in ("hidden_size", "dim", "d_model"):
                if key in mapping:
                    return int(mapping[key])
            continue
        for key in ("hidden_size", "dim", "d_model"):
            value = getattr(obj, key, None)
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, dict) and "hidden_size" in value:
                return int(value["hidden_size"])
    weight = getattr(embed, "weight", None)
    if weight is not None:
        shape = getattr(weight, "shape", None)
        if shape:
            return int(shape[-1])
    raise RuntimeError("could not infer MLX d_model")


def mlx_repo_id(cfg: AppConfig) -> str:
    return cfg.model.mlx_repo or cfg.model.name


def load_mlx_model(cfg: AppConfig) -> tuple[MlxLensModel, TokenizerAdapter, str]:
    try:
        import mlx.core as mx  # noqa: F401
    except Exception as exc:
        raise MissingMLXError(
            "mlx is not installed. On Apple Silicon: uv sync --extra apple"
        ) from exc

    repo = mlx_repo_id(cfg)
    tokenizer_config: dict[str, Any] = {}
    if cfg.model.trust_remote_code:
        tokenizer_config["trust_remote_code"] = True

    errors: list[str] = []
    mlx_model = None
    tokenizer = None
    library = "mlx_lm"
    try:
        from mlx_lm import load as mlx_lm_load

        load_kwargs: dict[str, Any] = {}
        if tokenizer_config:
            load_kwargs["tokenizer_config"] = tokenizer_config
        if cfg.model.revision:
            load_kwargs["revision"] = cfg.model.revision
        mlx_model, tokenizer = mlx_lm_load(repo, **load_kwargs)
        library = "mlx_lm"
    except Exception as exc:
        errors.append(f"mlx_lm.load({repo!r}): {exc}")
        try:
            from mlx_vlm import load as mlx_vlm_load

            mlx_model, tokenizer = mlx_vlm_load(repo)
            library = "mlx_vlm"
        except Exception as vlm_exc:
            errors.append(f"mlx_vlm.load({repo!r}): {vlm_exc}")
            raise MissingMLXError(
                "failed to load an MLX model for "
                f"{repo!r}. Tried mlx-lm then mlx-vlm. "
                + " | ".join(errors)
            ) from vlm_exc

    adapter = TokenizerAdapter(tokenizer)
    if adapter.pad_token_id is None and adapter.eos_token_id is not None:
        try:
            adapter.pad_token = getattr(tokenizer, "eos_token", None)
        except Exception:
            pass
    lens_model = MlxLensModel(mlx_model, adapter, library=library)
    print(
        f"[load_model] {repo} backend=mlx library={library} "
        f"n_layers={lens_model.n_layers} d_model={lens_model.d_model}"
    )
    return lens_model, adapter, library
