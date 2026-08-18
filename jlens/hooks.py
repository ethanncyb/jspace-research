# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Forward-hook context manager for capturing the residual stream."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import torch
from torch import nn


class ActivationRecorder:
    """Captures residual-stream tensors at the given block indices.

    Registers a forward hook on each requested block on ``__enter__`` and
    removes them on ``__exit__``. On the next forward pass each block's output
    is stored in :attr:`activations`, keyed by block index. Stored tensors are
    not detached, so they can be passed straight to :func:`torch.autograd.grad`.

    Args:
        blocks: The sequence of residual blocks (e.g. ``model.layers``).
        at: Block indices to record at.
        start_graph_at: If given, the captured tensor at this index is marked
            ``requires_grad_(True)`` before downstream blocks see it. When the
            model's parameters all have ``requires_grad=False``, this makes the
            captured residual the leaf that roots the autograd graph, so the
            retained graph spans only this block onward.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        start_graph_at: int | None = None,
    ) -> None:
        self._blocks = blocks
        self._indices = sorted(set(at))
        self._start_graph_at = start_graph_at
        if start_graph_at is not None and start_graph_at not in self._indices:
            self._indices = sorted({*self._indices, start_graph_at})
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, index: int) -> Callable[..., None]:
        is_graph_root = index == self._start_graph_at

        def hook(module: nn.Module, inputs, output) -> None:
            # Some HF blocks return a tuple (hidden, present_kv, ...).
            tensor = output if torch.is_tensor(output) else output[0]
            if is_graph_root:
                tensor.requires_grad_(True)
            self.activations[index] = tensor

        return hook

    def __enter__(self) -> ActivationRecorder:
        try:
            for index in self._indices:
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(index))
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class ActivationSteerer:
    """Adds precomputed residual-stream deltas at selected block outputs.

    ``deltas[layer]`` may be a single ``[d_model]`` vector, shared across all
    selected positions, or ``[n_positions, d_model]`` for position-specific
    clean-norm scaling.  Hooks return the same broad output shape as the
    block: tensors remain tensors and tuples preserve every non-hidden item.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        *,
        deltas: Mapping[int, torch.Tensor],
        positions: Sequence[int],
    ) -> None:
        self._blocks = blocks
        self._deltas = dict(deltas)
        self._positions = list(positions)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int) -> Callable[..., torch.Tensor | tuple]:
        def hook(module: nn.Module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            changed = hidden.clone()
            delta = self._deltas[layer].to(hidden.device, hidden.dtype)
            if delta.ndim == 1:
                delta = delta.unsqueeze(0)
            if delta.shape[0] not in (1, len(self._positions)):
                raise ValueError(
                    f"layer {layer} delta has {delta.shape[0]} rows; expected 1 "
                    f"or {len(self._positions)}"
                )
            changed[:, self._positions, :] += delta.unsqueeze(0)
            if torch.is_tensor(output):
                return changed
            return (changed, *output[1:])

        return hook

    def __enter__(self) -> ActivationSteerer:
        try:
            for layer in sorted(self._deltas):
                self._handles.append(
                    self._blocks[layer].register_forward_hook(self._make_hook(layer))
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class GenerateActivationSteerer:
    """Adds a residual-stream delta during HuggingFace ``generate`` forwards.

    Unlike :class:`ActivationSteerer`, this handles both the prompt pass
    (``seq_len >= prompt_len``) and cached incremental steps (``seq_len``
    typically 1).  ``deltas[layer]`` is a single ``[d_model]`` (or
    ``[1, d_model]``) vector: the last-prompt-token scaled J-space direction.

    ``decode_mode="prompt_pass"`` writes that vector only at index
    ``prompt_len - 1`` on full-sequence forwards and leaves cached decode
    steps unchanged.  ``decode_mode="every_step"`` also writes it at the
    current last position on each decode step; on an uncached growing
    sequence it writes every index from ``prompt_len - 1`` through the end.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        *,
        deltas: Mapping[int, torch.Tensor],
        prompt_len: int,
        decode_mode: Literal["prompt_pass", "every_step"] = "prompt_pass",
    ) -> None:
        if prompt_len < 1:
            raise ValueError("prompt_len must be >= 1")
        if decode_mode not in ("prompt_pass", "every_step"):
            raise ValueError("decode_mode must be 'prompt_pass' or 'every_step'")
        self._blocks = blocks
        self._deltas = dict(deltas)
        self._prompt_len = prompt_len
        self._decode_mode = decode_mode
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _vector(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        delta = self._deltas[layer].to(hidden.device, hidden.dtype)
        if delta.ndim == 2:
            if delta.shape[0] != 1:
                raise ValueError(
                    f"layer {layer} delta has {delta.shape[0]} rows; "
                    "generate steering expects a single [d_model] vector"
                )
            delta = delta[0]
        elif delta.ndim != 1:
            raise ValueError(
                f"layer {layer} delta must have shape [d_model] or "
                f"[1, d_model]; got {tuple(delta.shape)}"
            )
        return delta

    def _positions(self, seq_len: int) -> list[int] | None:
        if seq_len >= self._prompt_len:
            if self._decode_mode == "prompt_pass":
                return [self._prompt_len - 1]
            return list(range(self._prompt_len - 1, seq_len))
        if self._decode_mode == "every_step":
            return [seq_len - 1]
        return None

    def _make_hook(self, layer: int) -> Callable[..., torch.Tensor | tuple | None]:
        def hook(module: nn.Module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            positions = self._positions(hidden.shape[1])
            if positions is None:
                return None
            changed = hidden.clone()
            delta = self._vector(layer, hidden)
            changed[:, positions, :] += delta
            if torch.is_tensor(output):
                return changed
            return (changed, *output[1:])

        return hook

    def __enter__(self) -> GenerateActivationSteerer:
        try:
            for layer in sorted(self._deltas):
                self._handles.append(
                    self._blocks[layer].register_forward_hook(self._make_hook(layer))
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
