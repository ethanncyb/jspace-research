"""J-Space mean_replace intervention."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from gsm8k_jspace.config import InterventionSection


class MeanReplaceController:
    def __init__(
        self,
        jlens,
        lens_model,
        layers: list[int],
        spec: InterventionSection,
        *,
        compute_device: str = "cpu",
        log_path: Path | None = None,
    ) -> None:
        if not 0.0 <= spec.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        self._jlens = jlens
        self._blocks = lens_model.layers
        self.layers = sorted(layers)
        self.spec = spec
        self.compute_device = compute_device
        self.log_path = Path(log_path) if log_path else None
        self._handles: list = []
        self._log_fh = None
        self._run_sum: dict[int, torch.Tensor] = {}
        self._run_count: dict[int, int] = {}
        self.stats: dict[int, list[float]] = {
            layer: [0, 0, 0.0, 0.0] for layer in self.layers
        }
        self._calls = 0

    def reset_example(self, example_id: str, prompt_length: int) -> None:
        self._run_sum = {}
        self._run_count = {}

    def before_generation(self) -> None:
        return None

    def after_generation(self) -> None:
        return None

    def _feature_indices(self, z: torch.Tensor) -> torch.Tensor:
        mode = self.spec.features.mode
        k = min(int(self.spec.features.top_k), z.shape[-1])
        if mode == "top_abs":
            return z.abs().topk(k, dim=-1).indices
        if mode == "explicit":
            idx = torch.tensor(self.spec.features.indices, device=z.device)
            return idx.unsqueeze(0).expand(z.shape[0], -1)
        if mode == "random_matched":
            g = torch.Generator(device="cpu")
            g.manual_seed(self.spec.features.random_seed + self._calls)
            perm = torch.randperm(z.shape[-1], generator=g)[:k]
            return perm.to(z.device).unsqueeze(0).expand(z.shape[0], -1)
        raise ValueError(f"unknown feature mode {mode!r}")

    def apply_intervention(self, z: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.spec.method == "none" or self.spec.strength == 0.0:
            return z
        topk_idx = self._feature_indices(z)
        if z.shape[0] > 1:
            mean = z.mean(dim=0, keepdim=True).expand_as(z)
        else:
            count = self._run_count.get(layer_idx, 0)
            if count == 0:
                return z
            mean = (self._run_sum[layer_idx] / count).to(z.device).expand_as(z)
        target = z + self.spec.strength * (mean - z)
        return z.scatter(-1, topk_idx, target.gather(-1, topk_idx))

    def _update_running_mean(self, z: torch.Tensor, layer_idx: int) -> None:
        summed = z.sum(dim=0).detach().cpu()
        if layer_idx in self._run_sum:
            self._run_sum[layer_idx] += summed
            self._run_count[layer_idx] += z.shape[0]
        else:
            self._run_sum[layer_idx] = summed
            self._run_count[layer_idx] = z.shape[0]

    def hook_fn(self, layer_idx: int):
        def hook(module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            sel = hidden[0]
            z = self._jlens.project_to_jspace(sel, layer_idx)
            z_new = self.apply_intervention(z, layer_idx)
            self._update_running_mean(z, layer_idx)
            dz = z_new - z
            z_norm = torch.linalg.vector_norm(z)
            rel_z = (torch.linalg.vector_norm(dz) / z_norm).item() if z_norm > 0 else 0.0
            if self.spec.method == "none" or self.spec.strength == 0.0:
                delta_h = torch.zeros_like(sel)
            else:
                delta_h = self._jlens.project_from_jspace(
                    dz, layer_idx, compute_device=self.compute_device
                )
            h_norm = torch.linalg.vector_norm(sel.float())
            rel_h = (
                (torch.linalg.vector_norm(delta_h) / h_norm).item() if h_norm > 0 else 0.0
            )
            changed = hidden.clone()
            changed[0] += delta_h.to(changed.dtype)
            stats = self.stats[layer_idx]
            stats[0] += 1
            stats[1] += sel.shape[0]
            stats[2] += rel_z
            stats[3] += rel_h
            if self._log_fh is not None and self._calls < 200:
                self._log_fh.write(
                    json.dumps(
                        {
                            "layer": layer_idx,
                            "seq_len": int(hidden.shape[1]),
                            "rel_change_jspace": rel_z,
                            "rel_change_hidden": rel_h,
                            "method": self.spec.method,
                            "strength": self.spec.strength,
                        }
                    )
                    + "\n"
                )
            self._calls += 1
            if torch.is_tensor(output):
                return changed
            return (changed, *output[1:])

        return hook

    def __enter__(self) -> "MeanReplaceController":
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("w")
        for layer in self.layers:
            self._handles.append(self._blocks[layer].register_forward_hook(self.hook_fn(layer)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.spec.method,
            "top_k": self.spec.features.top_k,
            "strength": self.spec.strength,
            "feature_mode": self.spec.features.mode,
            "layers": self.layers,
            "per_layer": {
                str(layer): {
                    "n_forwards": int(s[0]),
                    "n_positions_modified": int(s[1]),
                    "mean_rel_change_jspace": s[2] / s[0] if s[0] else 0.0,
                    "mean_rel_change_hidden": s[3] / s[0] if s[0] else 0.0,
                }
                for layer, s in self.stats.items()
            },
        }
