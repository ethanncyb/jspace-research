"""Exact no-op controller: hooks fire but the original output object is returned."""

from __future__ import annotations

from typing import Any


class NoOpController:
    def __init__(self, lens_model, layers: list[int]) -> None:
        self._blocks = lens_model.layers
        self.layers = sorted(layers)
        self._handles: list = []
        self._n_calls = 0

    def reset_example(self, example_id: str, prompt_length: int) -> None:
        self._n_calls = 0

    def before_generation(self) -> None:
        return None

    def after_generation(self) -> None:
        return None

    def hook_fn(self, layer_idx: int):
        def hook(module, inputs, output):
            self._n_calls += 1
            return output

        return hook

    def __enter__(self) -> "NoOpController":
        for layer in self.layers:
            self._handles.append(self._blocks[layer].register_forward_hook(self.hook_fn(layer)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def summary(self) -> dict[str, Any]:
        return {
            "method": "none",
            "layers": self.layers,
            "n_hook_calls": self._n_calls,
            "delta_hidden_norm": 0.0,
            "delta_jspace_norm": 0.0,
        }
