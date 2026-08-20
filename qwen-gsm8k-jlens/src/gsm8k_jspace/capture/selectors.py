"""Layer and token selector resolution."""

from __future__ import annotations

from gsm8k_jspace.config import LayerSelector, TokenSelector


def resolve_layers(
    spec: LayerSelector, n_layers: int, fitted_layers: list[int]
) -> list[int]:
    fitted = set(fitted_layers)
    if spec.mode == "late":
        start = n_layers // 3
        stop = max(start + 1, n_layers - max(2, n_layers // 6))
        layers = [layer for layer in range(start, stop) if layer in fitted]
        if not layers:
            raise ValueError("late band has no overlap with fitted lens layers")
        return layers
    if spec.mode == "all_fitted":
        return sorted(fitted)
    if spec.mode == "range":
        start = int(spec.start)
        stop = int(spec.stop)
        stride = int(spec.stride)
        layers = [layer for layer in range(start, stop, stride) if layer in fitted]
        return sorted(set(layers))
    layers = sorted({int(layer) for layer in spec.values})
    unknown = sorted(set(layers) - fitted)
    if unknown:
        raise ValueError(
            f"layers {unknown} not fitted in the lens (fitted: {sorted(fitted)})"
        )
    return layers


def should_keep_position(
    selector: TokenSelector,
    *,
    call_index: int,
    generated_position: int | None,
    is_prefill: bool,
) -> bool:
    mode = selector.mode
    if mode == "prompt_last":
        return is_prefill
    if mode == "all_generated":
        return not is_prefill or selector.include_prompt
    if mode == "generated_stride":
        if is_prefill:
            return selector.include_prompt
        if generated_position is None:
            return False
        return generated_position % selector.stride == 0
    if mode == "generated_last":
        return False
    if mode == "full_sequence":
        return False
    if mode == "word_end":
        return not is_prefill or selector.include_prompt
    if mode == "explicit":
        if generated_position is None:
            return is_prefill and selector.include_prompt
        return generated_position in set(selector.positions)
    raise ValueError(f"unknown token mode {mode!r}")
