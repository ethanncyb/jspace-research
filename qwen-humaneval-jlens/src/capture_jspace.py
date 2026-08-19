"""Read-only J-Space activation capture during generation (Phase 1).

Hooks record, for every hooked layer and every generation step, what the
J-Lens reads out of the residual stream. Hooks NEVER modify or return
activations — this module is observation-only by design.

Per (layer, generation step) record:
  * layer index, absolute token position, and (attached by the runner) the
    generated token whose activation this is;
  * hidden-state L2 norm;
  * J-Space activation L2 norm (``|J_l @ h|``);
  * top-k J-Space token readouts — ``topk(unembed(J_l @ h))`` — the lens's
    native "what is this activation disposed to say" view.

---------------------------------------------------------------------------
TODO(Phase 2) — intervention lives here later, as a sibling context manager:
  * zero_topk:      zero the k largest-|z| J-Space coordinates per position
  * mean_replace:   replace top-k coordinates with the across-position mean
  * subtract_mean:  z' = z - mean_over_positions(z)
  * project back:   h' = h + pinv(J_l) @ (z' - z)   (JSpaceProjector)
  * rerun HumanEval with the hook active and compare pass@1 vs baseline
---------------------------------------------------------------------------
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch


def resolve_layers(spec, n_layers: int, fitted_layers: list[int]) -> list[int]:
    """``"late"`` -> the jlens late band (n//3 .. n - max(2, n//6)) intersected
    with fitted layers; an explicit list is validated and returned sorted."""
    fitted = set(fitted_layers)
    if spec == "late":
        start = n_layers // 3
        stop = max(start + 1, n_layers - max(2, n_layers // 6))
        layers = [layer for layer in range(start, stop) if layer in fitted]
        if not layers:
            raise ValueError("late band has no overlap with fitted lens layers")
        return layers
    layers = sorted(set(int(layer) for layer in spec))
    unknown = sorted(set(layers) - fitted)
    if unknown:
        raise ValueError(
            f"layers {unknown} not fitted in the lens (fitted: {sorted(fitted)})"
        )
    return layers


class JSpaceCapture:
    """Context manager registering observation-only forward hooks.

    One instance per task: ``prompt_len`` is needed to convert the per-call
    sequence offsets into absolute token positions. Records accumulate in
    :attr:`records`; the runner attaches generated-token info and writes them
    via :meth:`save`.
    """

    def __init__(
        self,
        lens_model,
        jlens_iface,
        *,
        layers: list[int],
        top_k: int,
        prompt_len: int,
    ) -> None:
        self._blocks = lens_model.layers
        self._lens_model = lens_model
        self._jlens = jlens_iface
        self.layers = sorted(layers)
        self.top_k = int(top_k)
        self._prompt_len = int(prompt_len)
        self._call_index = 0
        self._handles: list = []
        self.records: list[dict] = []

    def _make_hook(self, layer: int):
        def hook(module, inputs, output) -> None:
            hidden = output if torch.is_tensor(output) else output[0]
            # hidden: [batch, seq_len, d_model]; observe the LAST position only.
            # Prefill (call 0): last prompt token at position prompt_len-1.
            # Decode call c>=1: the newly fed token at position prompt_len-1+c.
            last = hidden[0, -1].detach()
            position = self._prompt_len - 1 + self._call_index

            z = self._jlens.project_to_jspace(last, layer)
            record = {
                "layer": layer,
                "call_index": self._call_index,
                "position": position,
                "hidden_norm": torch.linalg.vector_norm(last.float()).item(),
                "jspace_norm": torch.linalg.vector_norm(z.float()).item(),
                "top_jspace_tokens": [
                    [tok, round(logit, 4)]
                    for tok, logit in self._jlens.top_jspace_tokens(
                        last, layer, self._lens_model, self.top_k
                    )
                ],
            }
            self.records.append(record)
            # call_index advances on the last hooked layer's hook (hooks fire
            # in layer order within one forward pass).
            if layer == self.layers[-1]:
                self._call_index += 1

        return hook

    def __enter__(self) -> "JSpaceCapture":
        for layer in self.layers:
            self._handles.append(
                self._blocks[layer].register_forward_hook(self._make_hook(layer))
            )
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def attach_tokens(self, generated_ids: list[int], tokenizer) -> None:
        """Associate each decode-step record with the generated token whose
        activation it observed (call c>=1 saw generated token ``c-1``; the
        prefill call 0 saw the last prompt token)."""
        for record in self.records:
            call = record["call_index"]
            if call == 0:
                record["token"] = "<prompt-last>"
                record["token_id"] = None
            elif call - 1 < len(generated_ids):
                token_id = generated_ids[call - 1]
                record["token_id"] = int(token_id)
                record["token"] = tokenizer.decode([token_id])
            else:
                record["token"] = None
                record["token_id"] = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(str(path) + ".gz", "wt") as fh:
            for record in self.records:
                fh.write(json.dumps(record) + "\n")
