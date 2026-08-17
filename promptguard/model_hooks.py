"""Qwen/Hugging Face loading and residual-stream READ/WRITE hooks.

The hook manager deliberately operates on whole decoder block outputs. Qwen
3.5's linear-attention (GDN) blocks may be observed, but writes are rejected
unless the layer is listed as a full-attention checkpoint.
"""

from __future__ import annotations

import contextlib
import enum
import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from promptguard.config import ModelConfig


class HookMode(str, enum.Enum):
    READ = "read"
    WRITE = "write"


Writer = Callable[[int, torch.Tensor], torch.Tensor]


def _resolve_attr(obj: Any, path: str) -> Any:
    return functools.reduce(getattr, path.split("."), obj)


def _text_config(model: nn.Module) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model has no Hugging Face-style config")
    get_text_config = getattr(config, "get_text_config", None)
    return get_text_config() if get_text_config is not None else config


def _find_layer_path(model: nn.Module, explicit: str | None = None) -> str:
    candidates = [explicit] if explicit else []
    candidates += [
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]
    for path in candidates:
        if not path:
            continue
        try:
            layers = _resolve_attr(model, path)
        except AttributeError:
            continue
        if isinstance(layers, Sequence) or isinstance(layers, nn.ModuleList):
            return path
    raise ValueError("could not locate decoder blocks; set model.layer_path in config")


def discover_full_attention_layers(
    model: nn.Module, fallback: Iterable[int]
) -> list[int]:
    """Use ``config.layer_types`` when present, otherwise return ``fallback``."""

    layer_types = getattr(_text_config(model), "layer_types", None)
    if layer_types:
        found = [
            index
            for index, layer_type in enumerate(layer_types)
            if str(layer_type) in {"full_attention", "attention"}
        ]
        if found:
            return found
    return sorted(set(fallback))


def _torch_dtype(name: str) -> torch.dtype:
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


@dataclass
class LoadedModel:
    model: nn.Module
    tokenizer: Any
    hooks: HookedModel


def load_model(config: ModelConfig) -> LoadedModel:
    """Load a text-capable Qwen model and construct its hook adapter.

    Multimodal Qwen 3.5 checkpoints are loaded through
    ``AutoModelForImageTextToText``. Text-only checkpoints use
    ``AutoModelForCausalLM``. No model is downloaded until this function is
    called, keeping all other modules and tests lightweight.
    """

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    hf_config = AutoConfig.from_pretrained(
        config.name, trust_remote_code=config.trust_remote_code
    )
    load_kwargs: dict[str, Any] = {
        "dtype": _torch_dtype(config.dtype),
        "trust_remote_code": config.trust_remote_code,
    }
    if config.device_map is not None:
        load_kwargs["device_map"] = config.device_map

    if getattr(hf_config, "vision_config", None) is not None:
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                config.name, **load_kwargs
            )
        except ImportError as exc:
            raise RuntimeError(
                "this Qwen checkpoint is multimodal; install a Transformers "
                "version providing AutoModelForImageTextToText"
            ) from exc
    else:
        model = AutoModelForCausalLM.from_pretrained(config.name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        config.name, trust_remote_code=config.trust_remote_code
    )
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        hooks=HookedModel(
            model,
            tokenizer,
            layer_indices=config.layer_indices,
            hidden_dim=config.hidden_dim,
            layer_path=config.layer_path,
            max_length=config.max_length,
        ),
    )


class ActivationHooks:
    """Context manager that captures and optionally rewrites block outputs."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        indices: Iterable[int],
        *,
        mode: HookMode = HookMode.READ,
        writer: Writer | None = None,
        writable_layers: Iterable[int] = (),
        detach: bool = True,
        to_cpu: bool = True,
    ) -> None:
        self.layers = layers
        self.indices = sorted(set(indices))
        self.mode = HookMode(mode)
        self.writer = writer
        self.writable_layers = set(writable_layers)
        self.detach = detach
        self.to_cpu = to_cpu
        self.activations: dict[int, torch.Tensor] = {}
        self.modified_activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        if self.mode is HookMode.WRITE:
            if writer is None:
                raise ValueError("WRITE mode requires a writer callback")
            forbidden = set(self.indices) - self.writable_layers
            if forbidden:
                raise ValueError(
                    f"writes are only allowed at full-attention layers; got "
                    f"read-only layers {sorted(forbidden)}"
                )

    def _stored(self, tensor: torch.Tensor) -> torch.Tensor:
        stored = tensor.detach() if self.detach else tensor
        return stored.cpu() if self.to_cpu else stored

    def _hook(self, index: int):
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
            hidden = output if torch.is_tensor(output) else output[0]
            self.activations[index] = self._stored(hidden)
            if self.mode is HookMode.READ:
                return None
            assert self.writer is not None
            changed = self.writer(index, hidden)
            if changed.shape != hidden.shape:
                raise ValueError(
                    f"writer changed layer {index} shape from {tuple(hidden.shape)} "
                    f"to {tuple(changed.shape)}"
                )
            self.modified_activations[index] = self._stored(changed)
            if torch.is_tensor(output):
                return changed
            if isinstance(output, tuple):
                return (changed, *output[1:])
            raise TypeError(f"unsupported block output type: {type(output).__name__}")

        return capture

    def __enter__(self) -> ActivationHooks:
        try:
            for index in self.indices:
                self._handles.append(
                    self.layers[index].register_forward_hook(self._hook(index))
                )
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, *_exc: object) -> None:
        self.close()


class HookedModel:
    """Thin adapter exposing capture and write contexts around an HF model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        layer_indices: Iterable[int],
        hidden_dim: int | None = None,
        layer_path: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer_path = _find_layer_path(model, layer_path)
        self.layers: Sequence[nn.Module] = _resolve_attr(model, self.layer_path)
        self.full_attention_layers = discover_full_attention_layers(
            model, layer_indices
        )
        config_dim = int(_text_config(model).hidden_size)
        if hidden_dim is not None and hidden_dim != config_dim:
            raise ValueError(
                f"configured hidden_dim={hidden_dim}, model reports {config_dim}"
            )
        self.hidden_dim = config_dim
        self.max_length = max_length
        invalid = [
            i for i in self.full_attention_layers if not 0 <= i < len(self.layers)
        ]
        if invalid:
            raise IndexError(f"hook layer indices outside decoder: {invalid}")

    @property
    def input_device(self) -> torch.device:
        try:
            embedding = self.model.get_input_embeddings()
            return embedding.weight.device
        except (AttributeError, NotImplementedError):
            return next(self.model.parameters()).device

    def hook_context(
        self,
        *,
        mode: HookMode = HookMode.READ,
        indices: Iterable[int] | None = None,
        writer: Writer | None = None,
        detach: bool = True,
        to_cpu: bool = True,
    ) -> ActivationHooks:
        return ActivationHooks(
            self.layers,
            self.full_attention_layers if indices is None else indices,
            mode=mode,
            writer=writer,
            writable_layers=self.full_attention_layers,
            detach=detach,
            to_cpu=to_cpu,
        )

    def tokenize(self, texts: str | list[str]) -> Mapping[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=isinstance(texts, list),
            truncation=True,
            max_length=self.max_length,
        )
        return {key: value.to(self.input_device) for key, value in encoded.items()}

    @torch.inference_mode()
    def capture(
        self, texts: str | list[str], *, indices: Iterable[int] | None = None
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        encoded = self.tokenize(texts)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(encoded["input_ids"])
        with self.hook_context(indices=indices) as hooks:
            self.model(**encoded, use_cache=False)
        return hooks.activations, attention_mask.detach().cpu()

    def generate(
        self,
        text: str,
        *,
        writer: Writer | None = None,
        max_new_tokens: int = 64,
        **generation_kwargs: Any,
    ) -> str:
        encoded = self.tokenize(text)
        context = (
            self.hook_context(mode=HookMode.WRITE, writer=writer, to_cpu=False)
            if writer is not None
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), context:
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
        prompt_length = encoded["input_ids"].shape[1]
        return self.tokenizer.decode(
            output_ids[0, prompt_length:], skip_special_tokens=True
        )
