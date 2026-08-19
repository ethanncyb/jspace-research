"""J-Space intervention hooks (Phase 2).

At each hooked layer the hook performs, on every position of every forward
pass (prompt prefill and each decode step):

    h  ->  z = J_l @ h              (project into J-Space)
    z  ->  z' = method(z)           (intervention)
    h' =  h + pinv(J_l) @ (z' - z)  (transport only the delta back)

Because only the J-Space *delta* is transported back, ``method="none"`` is an
exact no-op and must reproduce the same-hardware baseline completions
bit-for-bit — this is the correctness gate for the whole pipeline.

Methods implemented now
-----------------------
none          pass-through control (delta is exactly zero)
mean_replace  move the ``top_k`` largest-|z| coordinates of each position
              toward a mean J-Space activation by ``strength``:
              ``z' = z + strength * (mean - z)`` on those coordinates.
              strength=1.0 is full replacement — measured to be model-breaking
              (hidden-space delta ≈ 3x ||h|| per layer, due to massive
              activations + pinv amplification); use small strength values.
              At prefill the mean is taken across the positions of the current
              forward; at decode (one position per forward) it is the running
              mean over all positions seen so far in the current task,
              excluding the current one.

TODO(future ablations — deliberately not implemented yet)
---------------------------------------------------------
zero_topk        zero the top-k coordinates instead of mean-replacing them
subtract_mean    z' = z - running_mean(z) (remove the shared component)
random_ablation  replace k random (not top-k) coordinates, matched magnitude
calibration_mean mean from a held-out calibration set instead of the same task
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

METHODS = ("mean_replace", "none")

# Per-forward detail lines are sampled; aggregate stats cover every call.
_MAX_DETAILED_CALLS = 200


class JSPaceIntervention:
    """Context manager registering J-Space intervention hooks.

    One instance per run; call :meth:`reset` between tasks so the running
    mean is task-local.

    Args:
        jlens: :class:`src.load_jlens.JLens` (projections + pinv).
        lens_model: ``jlens.HFLensModel`` (provides ``layers`` / ``n_layers``).
        layers: resolved layer indices to hook (see capture.resolve_layers).
        method: one of :data:`METHODS`.
        top_k: number of largest-|z| coordinates replaced per position.
        token_strategy: ``"all_positions"`` — every position of every forward
            during generation (prefill + decode). ``"all_generated_tokens"``
            is accepted as a legacy alias.
        log_path: optional JSONL file for sampled per-forward detail records.
    """

    def __init__(
        self,
        jlens,
        lens_model,
        layers: list[int],
        method: str = "mean_replace",
        top_k: int = 50,
        strength: float = 1.0,
        token_strategy: str = "all_positions",
        log_path: str | Path | None = None,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        # "all_generated_tokens" is the legacy name for the same semantics
        if token_strategy == "all_generated_tokens":
            token_strategy = "all_positions"
        if token_strategy != "all_positions":
            raise ValueError("only 'all_positions' is implemented")
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        self._jlens = jlens
        self._blocks = lens_model.layers
        self.layers = sorted(layers)
        self.method = method
        self.top_k = int(top_k)
        self.strength = float(strength)
        self.token_strategy = token_strategy
        self.log_path = Path(log_path) if log_path else None
        self._handles: list = []
        self._log_fh = None
        self._calls = 0
        # running per-dimension mean of z, per layer, for the current task
        self._run_sum: dict[int, torch.Tensor] = {}
        self._run_count: dict[int, int] = {}
        # per-layer aggregates across the run:
        # [n_forwards, n_positions, sum_rel_z, sum_rel_h]
        self.stats: dict[int, list[float]] = {layer: [0, 0, 0.0, 0.0] for layer in self.layers}

    # -- task lifecycle -------------------------------------------------------

    def reset(self) -> None:
        """Clear the running-mean buffers; call before each new task."""
        self._run_sum = {}
        self._run_count = {}

    # -- intervention math ------------------------------------------------------

    def project_to_jspace(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self._jlens.project_to_jspace(hidden_states, layer_idx)

    def apply_intervention(
        self, jspace_states: torch.Tensor, layer_idx: int, token_idx: int = -1
    ) -> torch.Tensor:
        """``z' = method(z)`` for z of shape ``[n_positions, d_model]``.

        ``token_idx`` is informational (absolute position of the last token);
        the method itself is position-agnostic.
        """
        z = jspace_states
        if self.method == "none":
            return z
        if self.method == "mean_replace":
            topk_idx = z.abs().topk(min(self.top_k, z.shape[-1]), dim=-1).indices
            if z.shape[0] > 1:
                # prefill: mean across the positions of this forward
                mean = z.mean(dim=0, keepdim=True).expand_as(z)
            else:
                # decode: running mean over previous positions of this task
                count = self._run_count.get(layer_idx, 0)
                if count == 0:
                    return z  # no history yet; leave the first position intact
                mean = (self._run_sum[layer_idx] / count).to(z.device).expand_as(z)
            target = z + self.strength * (mean - z)
            return z.scatter(-1, topk_idx, target.gather(-1, topk_idx))
        raise AssertionError("unreachable")

    def project_back(self, modified_jspace_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self._jlens.project_from_jspace(modified_jspace_states, layer_idx)

    def _update_running_mean(self, z: torch.Tensor, layer_idx: int) -> None:
        """Fold the ORIGINAL (pre-intervention) z of this forward into the
        task-local running mean."""
        s = z.sum(dim=0).detach().cpu()
        if layer_idx in self._run_sum:
            self._run_sum[layer_idx] += s
            self._run_count[layer_idx] += z.shape[0]
        else:
            self._run_sum[layer_idx] = s
            self._run_count[layer_idx] = z.shape[0]

    # -- hooks --------------------------------------------------------------------

    def hook_fn(self, layer_idx: int):
        """Return the forward hook bound to ``layer_idx``."""

        def hook(module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            # hidden: [batch, seq_len, d_model]; intervene on all positions.
            sel = hidden[0]
            z = self.project_to_jspace(sel, layer_idx)
            z_new = self.apply_intervention(z, layer_idx)
            self._update_running_mean(z, layer_idx)

            dz = z_new - z
            z_norm = torch.linalg.vector_norm(z)
            rel_z = (torch.linalg.vector_norm(dz) / z_norm).item() if z_norm > 0 else 0.0
            if self.method == "none":
                delta_h = torch.zeros_like(sel)
            else:
                delta_h = self.project_back(dz, layer_idx)
            h_norm = torch.linalg.vector_norm(sel.float())
            rel_h = (
                (torch.linalg.vector_norm(delta_h) / h_norm).item()
                if h_norm > 0 else 0.0
            )

            changed = hidden.clone()
            changed[0] += delta_h.to(changed.dtype)

            stats = self.stats[layer_idx]
            stats[0] += 1
            stats[1] += sel.shape[0]
            stats[2] += rel_z
            stats[3] += rel_h
            if self._log_fh is not None and self._calls < _MAX_DETAILED_CALLS:
                self._log_fh.write(
                    json.dumps(
                        {
                            "layer": layer_idx,
                            "seq_len": int(hidden.shape[1]),
                            "n_positions": int(sel.shape[0]),
                            "method": self.method,
                            "top_k": self.top_k,
                            "strength": self.strength,
                            "token_strategy": self.token_strategy,
                            "rel_change_jspace": rel_z,
                            "rel_change_hidden": rel_h,
                        }
                    )
                    + "\n"
                )
            self._calls += 1

            if torch.is_tensor(output):
                return changed
            return (changed, *output[1:])

        return hook

    # -- context manager -----------------------------------------------------------

    def __enter__(self) -> "JSPaceIntervention":
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("w")
        for layer in self.layers:
            self._handles.append(
                self._blocks[layer].register_forward_hook(self.hook_fn(layer))
            )
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def summary(self) -> dict:
        return {
            "method": self.method,
            "top_k": self.top_k,
            "strength": self.strength,
            "token_strategy": self.token_strategy,
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
