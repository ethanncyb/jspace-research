"""Residual-stream capture and intervention hooks for Qwen 3.5 detectors.

The detector deliberately performs two forwards: one over a trusted prefix and
one over the exact prefix tokens followed by separately encoded untrusted text.
Hooks retain device tensors only for the duration of a forward and expose
detached float32 copies for training, logging, and diagnostics.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

DEFAULT_FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)


@dataclass(frozen=True)
class DevicePolicy:
    """Resolved inference device and dtype."""

    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class EncodedPair:
    """Tokenization with an exact boundary between trusted and appended text."""

    clean_ids: torch.Tensor
    full_ids: torch.Tensor
    prefix_length: int

    @property
    def appended_length(self) -> int:
        return int(self.full_ids.shape[-1] - self.prefix_length)


@dataclass
class LayerDiagnostic:
    layer: int
    l2_distance: float
    metadata: dict[str, Any]


Processor = Callable[
    [int, torch.Tensor, torch.Tensor, int],
    tuple[torch.Tensor | None, Mapping[str, Any] | None],
]


def select_device_dtype(
    device: str | torch.device = "auto", dtype: str | torch.dtype = "auto"
) -> DevicePolicy:
    """Select CUDA, MPS, or CPU and a conservative matching dtype."""

    if str(device) == "auto":
        if torch.cuda.is_available():
            resolved_device = torch.device("cuda")
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            resolved_device = torch.device("mps")
        else:
            resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(device)

    if isinstance(dtype, torch.dtype):
        resolved_dtype = dtype
    elif dtype == "auto":
        resolved_dtype = (
            torch.bfloat16
            if resolved_device.type == "cuda"
            else torch.float16
            if resolved_device.type == "mps"
            else torch.float32
        )
    else:
        try:
            resolved_dtype = getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"unknown torch dtype: {dtype}") from exc
        if not isinstance(resolved_dtype, torch.dtype):
            raise ValueError(f"unknown torch dtype: {dtype}")
    return DevicePolicy(resolved_device, resolved_dtype)


def load_model_and_tokenizer(
    model_id: str,
    *,
    device: str | torch.device = "auto",
    dtype: str | torch.dtype = "auto",
    **from_pretrained_kwargs: Any,
) -> tuple[nn.Module, Any, DevicePolicy]:
    """Load a HuggingFace causal LM and tokenizer using the device policy."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    policy = select_device_dtype(device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=policy.dtype, **from_pretrained_kwargs
    )
    model.to(policy.device).eval()
    tokenizer_keys = {
        "revision",
        "cache_dir",
        "token",
        "trust_remote_code",
        "local_files_only",
    }
    tokenizer_kwargs = {
        key: value
        for key, value in from_pretrained_kwargs.items()
        if key in tokenizer_keys
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    return model, tokenizer, policy


def _resolve_attr(obj: Any, path: str) -> Any:
    return functools.reduce(getattr, path.split("."), obj)


def resolve_text_decoder(model: nn.Module) -> nn.Module:
    """Find the text decoder without depending on one HF wrapper layout."""

    candidates = (
        "model.language_model",
        "language_model",
        "model",
        "transformer",
        "gpt_neox",
    )
    for path in candidates:
        with contextlib.suppress(AttributeError):
            decoder = _resolve_attr(model, path)
            if any(hasattr(decoder, name) for name in ("layers", "h")):
                return decoder
    if any(hasattr(model, name) for name in ("layers", "h")):
        return model
    raise ValueError(f"could not resolve text decoder for {type(model).__name__}")


def resolve_residual_stack(model: nn.Module) -> Sequence[nn.Module]:
    decoder = resolve_text_decoder(model)
    for name in ("layers", "h"):
        blocks = getattr(decoder, name, None)
        if blocks is not None:
            return blocks
    raise ValueError("resolved decoder does not expose a residual stack")


def get_text_config(model: nn.Module) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        return None
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return getattr(config, "text_config", config)


def hidden_size(model: nn.Module) -> int:
    config = get_text_config(model)
    for source in (config, model):
        for name in ("hidden_size", "d_model"):
            value = getattr(source, name, None)
            if value is not None:
                return int(value)
    raise ValueError("could not derive hidden size from model config")


def validate_layer_layout(
    model: nn.Module,
    requested: Iterable[int] = DEFAULT_FULL_ATTENTION_LAYERS,
) -> tuple[int, ...]:
    """Validate indices and, when present, Qwen's per-layer attention types."""

    layers = resolve_residual_stack(model)
    indices = tuple(int(i) for i in requested)
    invalid = [i for i in indices if i < 0 or i >= len(layers)]
    if invalid:
        raise ValueError(
            f"checkpoint layers out of range for {len(layers)} blocks: {invalid}"
        )

    config = get_text_config(model)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        if len(layer_types) != len(layers):
            raise ValueError(
                f"text_config.layer_types has {len(layer_types)} entries for "
                f"{len(layers)} residual blocks"
            )
        bad = {i: layer_types[i] for i in indices if layer_types[i] != "full_attention"}
        if bad:
            raise ValueError(
                f"requested checkpoints are not full-attention layers: {bad}"
            )
    return indices


def _token_ids(tokenizer: Any, text: str, *, special: bool) -> torch.Tensor:
    kwargs = {"return_tensors": "pt", "add_special_tokens": special}
    try:
        encoded = tokenizer(text, **kwargs)
    except TypeError:
        kwargs.pop("add_special_tokens")
        encoded = tokenizer(text, **kwargs)
        if not special:
            ids = encoded.input_ids
            bos = getattr(tokenizer, "bos_token_id", None)
            if bos is not None and ids.shape[-1] and int(ids[0, 0]) == bos:
                return ids[:, 1:]
    return encoded.input_ids


def encode_clean_and_appended(
    tokenizer: Any,
    clean_prompt: str,
    appended_text: str,
    *,
    appended_token_ids: Sequence[int] | torch.Tensor | None = None,
    max_length: int | None = None,
    device: torch.device | str | None = None,
) -> EncodedPair:
    """Encode each segment independently and concatenate token IDs exactly."""

    clean = _token_ids(tokenizer, clean_prompt, special=True)
    suffix = (
        _token_ids(tokenizer, appended_text, special=False)
        if appended_token_ids is None
        else torch.as_tensor(appended_token_ids, dtype=clean.dtype).reshape(1, -1)
    )
    if max_length is not None:
        if clean.shape[-1] > max_length:
            raise ValueError("trusted prefix alone exceeds maximum length")
        suffix = suffix[:, : max_length - clean.shape[-1]]
    if suffix.shape[-1] == 0:
        raise ValueError("appended text encoded to zero tokens")
    full = torch.cat((clean, suffix), dim=-1)
    if device is not None:
        clean, full = clean.to(device), full.to(device)
    return EncodedPair(clean, full, int(clean.shape[-1]))


def validate_candidate_prefix(
    tokenizer: Any, clean_prompt: str, candidate_prompt: str
) -> torch.Tensor:
    """Return candidate suffix IDs only when clean IDs are an exact prefix."""

    clean = _token_ids(tokenizer, clean_prompt, special=True)
    candidate = _token_ids(tokenizer, candidate_prompt, special=True)
    n = clean.shape[-1]
    if candidate.shape[-1] <= n or not torch.equal(candidate[:, :n], clean):
        raise ValueError(
            "candidate_prompt token IDs do not have clean_prompt as an exact prefix"
        )
    return candidate[:, n:]


class ResidualHookController:
    """Context-managed two-pass residual capture with optional intervention.

    Observation-only layers are always read-only. A ``processor`` can replace
    hidden states only at layers validated as full-attention checkpoints.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        checkpoint_layers: Iterable[int],
        *,
        observation_layers: Iterable[int] | None = None,
        processor: Processor | None = None,
    ) -> None:
        self.blocks = blocks
        self.checkpoint_layers = tuple(sorted(set(checkpoint_layers)))
        self.observation_layers = tuple(
            sorted(
                set(
                    observation_layers
                    if observation_layers is not None
                    else range(len(blocks))
                )
            )
        )
        all_indices = {*self.checkpoint_layers, *self.observation_layers}
        bad = [i for i in all_indices if i < 0 or i >= len(blocks)]
        if bad:
            raise ValueError(f"hook layers out of range: {sorted(bad)}")
        self.processor = processor
        self.phase = "idle"
        self.prefix_length = 0
        self.prefill_length: int | None = None
        self.clean_last: dict[int, torch.Tensor] = {}
        self.appended_last: dict[int, torch.Tensor] = {}
        self.deltas: dict[int, torch.Tensor] = {}
        self.diagnostics: dict[int, LayerDiagnostic] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _hidden(output: Any) -> torch.Tensor:
        hidden = output if torch.is_tensor(output) else output[0]
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise TypeError(
                "residual block output must contain a [batch, sequence, hidden] tensor"
            )
        return hidden

    @staticmethod
    def _replace(output: Any, hidden: torch.Tensor) -> Any:
        if torch.is_tensor(output):
            return hidden
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        if isinstance(output, list):
            return [hidden, *output[1:]]
        raise TypeError("cannot replace hidden state in unsupported block output")

    def begin_clean(self) -> None:
        self.phase = "clean"
        self.clean_last.clear()
        self.appended_last.clear()
        self.deltas.clear()
        self.diagnostics.clear()

    def begin_appended(
        self, prefix_length: int, *, prefill_length: int | None = None
    ) -> None:
        if not self.clean_last:
            raise RuntimeError("clean pass must be captured before appended pass")
        self.phase = "appended"
        self.prefix_length = int(prefix_length)
        self.prefill_length = prefill_length
        self.appended_last.clear()
        self.deltas.clear()
        self.diagnostics.clear()

    def disable(self) -> None:
        self.phase = "idle"

    def _make_hook(self, layer: int) -> Callable[..., Any]:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
            hidden = self._hidden(output)
            last = hidden[:, -1, :]
            if self.phase == "clean":
                self.clean_last[layer] = last.detach().float().cpu()
                return None
            if self.phase != "appended":
                return None
            if (
                self.prefill_length is not None
                and hidden.shape[1] != self.prefill_length
            ):
                return None

            self.appended_last[layer] = last.detach().float().cpu()
            if layer not in self.checkpoint_layers:
                return None  # GDN/observation-only layers cannot be mutated.
            if layer not in self.clean_last:
                raise RuntimeError(
                    f"clean activation missing for checkpoint layer {layer}"
                )
            baseline = self.clean_last[layer].to(last.device)
            delta = last.float() - baseline
            self.deltas[layer] = delta.detach().cpu()
            metadata: Mapping[str, Any] | None = None
            changed: torch.Tensor | None = None
            if self.processor is not None:
                changed, metadata = self.processor(
                    layer, hidden, delta, self.prefix_length
                )
            self.diagnostics[layer] = LayerDiagnostic(
                layer=layer,
                l2_distance=float(torch.linalg.vector_norm(delta).item()),
                metadata=dict(metadata or {}),
            )
            if changed is not None:
                if changed.shape != hidden.shape:
                    raise ValueError(
                        "processor replacement changed hidden tensor shape"
                    )
                return self._replace(output, changed)
            return None

        return hook

    def __enter__(self) -> ResidualHookController:
        try:
            for layer in sorted({*self.checkpoint_layers, *self.observation_layers}):
                self._handles.append(
                    self.blocks[layer].register_forward_hook(self._make_hook(layer))
                )
        except Exception:
            self._remove()
            raise
        return self

    def _remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, *_exc: Any) -> None:
        self._remove()


def run_paired_prefill(
    model: nn.Module,
    tokenizer: Any,
    clean_prompt: str,
    appended_text: str,
    controller: ResidualHookController,
    *,
    appended_token_ids: Sequence[int] | torch.Tensor | None = None,
    max_length: int | None = None,
) -> tuple[EncodedPair, Any]:
    """Execute clean and appended prefill passes through an installed controller."""

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    pair = encode_clean_and_appended(
        tokenizer,
        clean_prompt,
        appended_text,
        appended_token_ids=appended_token_ids,
        max_length=max_length,
        device=device,
    )
    controller.begin_clean()
    with torch.no_grad():
        model(input_ids=pair.clean_ids)
    controller.begin_appended(
        pair.prefix_length, prefill_length=pair.full_ids.shape[-1]
    )
    with torch.no_grad():
        output = model(input_ids=pair.full_ids)
    controller.disable()
    return pair, output
