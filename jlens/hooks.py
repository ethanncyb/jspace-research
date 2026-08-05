# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Forward-hook context manager for capturing the residual stream."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

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
