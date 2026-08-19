"""J-Lens / J-Space loading and the Phase-1 observation interface.

The Jacobian lens ("J-Lens") transports a residual-stream activation at layer
``l`` into the model's final-layer hidden basis via the averaged input-output
Jacobian ``J_l``; that final-layer basis is "J-Space". This module wraps
:class:`jlens.JacobianLens` behind the small interface the experiment uses:

* ``project_to_jspace(hidden_states, layer_idx)`` — real, ``h @ J_l.T``.
* ``project_from_jspace(jspace_states, layer_idx)`` — least-squares
  back-projection via the cached per-layer pseudo-inverse (Phase 2).
* ``get_supported_layers()`` / ``get_metadata()``.

If the fitted lens checkpoint cannot be loaded, :class:`MissingJLensError`
clearly names the missing dependency. ``jlens.source: identity`` in
config.yaml is a documented placeholder (J = I per layer) that keeps the
pipeline runnable for plumbing tests only — it is NOT a J-Lens, and the
placeholder status is recorded in run metadata and the final report.
"""

from __future__ import annotations

import torch

import jlens


class MissingJLensError(RuntimeError):
    """The fitted J-Lens checkpoint (the Phase-1 external dependency) could
    not be loaded."""


class JLens:
    """Observation-side J-Lens interface used by capture and analysis code."""

    def __init__(self, lens, *, source_desc: str, placeholder: bool = False) -> None:
        self._lens = lens
        self._pinv: dict[int, torch.Tensor] = {}
        self.placeholder = placeholder
        self._metadata = {
            "source": source_desc,
            "placeholder": placeholder,
            "status": "identity-placeholder" if placeholder else "fitted",
            "supported_layers": lens.source_layers,
            "n_layers_fitted": len(lens.source_layers),
            "d_model": lens.d_model,
            "n_prompts": lens.n_prompts,
        }

    # -- Phase-1 interface ---------------------------------------------------

    def project_to_jspace(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """``J_l @ h`` for ``hidden_states`` of shape ``[..., d_model]``."""
        return self._lens.transport(hidden_states.float(), layer_idx)

    def project_from_jspace(self, jspace_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Least-squares back-projection ``pinv(J_l) @ z``.

        ``J_l`` is an averaged Jacobian and not guaranteed invertible, so the
        back-projection uses the Moore–Penrose pseudo-inverse, precomputed
        once per layer in float32 and cached. Used by Phase-2 interventions;
        Phase-1 capture code never calls this.
        """
        if layer_idx not in self._pinv:
            J = self._lens.jacobians[layer_idx].float()
            self._pinv[layer_idx] = torch.linalg.pinv(J)
        pinv_T = self._pinv[layer_idx].T.to(jspace_states.device)
        return jspace_states.float() @ pinv_T

    def get_supported_layers(self) -> list[int]:
        return list(self._lens.source_layers)

    def get_metadata(self) -> dict:
        return dict(self._metadata)

    # -- convenience used by capture -----------------------------------------

    def top_jspace_tokens(self, hidden_state: torch.Tensor, layer_idx: int,
                          lens_model, k: int) -> list[tuple[str, float]]:
        """Top-k vocabulary readout of the J-Space projection of one position.

        ``hidden_state``: shape ``[d_model]``. Returns ``(token_str, logit)``
        pairs — the lens's native "what is this activation disposed to say".
        """
        z = self.project_to_jspace(hidden_state, layer_idx)
        logits = lens_model.unembed(z).float()
        top = logits.topk(k)
        tokens = [
            lens_model.tokenizer.decode([tid]) for tid in top.indices.tolist()
        ]
        return list(zip(tokens, top.values.tolist()))


def _identity_lens(n_layers: int, d_model: int):
    eye = torch.eye(d_model)
    return jlens.JacobianLens(
        {layer: eye.clone() for layer in range(n_layers)},
        n_prompts=0,
        d_model=d_model,
    )


def load_jlens(cfg: dict, hf_model, tokenizer):
    """Load the J-Lens per ``cfg["jlens"]`` and wrap the HF model.

    Returns ``(jlens_iface, lens_model)`` where ``lens_model`` is the
    :class:`jlens.HFLensModel` adapter (exposes ``layers`` / ``n_layers`` /
    ``unembed`` for the capture hooks).
    """
    lens_cfg = cfg["jlens"]
    lens_model = jlens.from_hf(hf_model, tokenizer)

    if lens_cfg["source"] == "identity":
        print("[load_jlens] WARNING: identity placeholder — NOT a fitted J-Lens; "
              "J-Space readouts are final-basis controls only")
        lens = _identity_lens(lens_model.n_layers, lens_model.d_model)
        return JLens(lens, source_desc="identity placeholder", placeholder=True), lens_model

    try:
        if lens_cfg["source"] == "hub":
            lens = jlens.JacobianLens.from_pretrained(
                lens_cfg["repo"],
                filename=lens_cfg["file"],
                revision=lens_cfg.get("revision"),
            )
            desc = f"{lens_cfg['repo']}@{lens_cfg.get('revision')}:{lens_cfg['file']}"
        elif lens_cfg["source"] == "local":
            lens = jlens.JacobianLens.from_pretrained(lens_cfg["local_path"])
            desc = str(lens_cfg["local_path"])
        else:
            raise MissingJLensError(f"unknown jlens source {lens_cfg['source']!r}")
    except Exception as exc:
        raise MissingJLensError(
            "MISSING DEPENDENCY: the fitted J-Lens checkpoint could not be "
            f"loaded (config: {lens_cfg}). An exact J-Lens fitted on "
            f"{cfg['model']['name']} is required for meaningful J-Space "
            "readouts; set jlens.source: identity only for plumbing/control "
            f"runs. Original error: {exc}"
        ) from exc

    if lens.d_model != lens_model.d_model:
        raise MissingJLensError(
            f"lens d_model={lens.d_model} != model d_model={lens_model.d_model}; "
            "this lens was not fitted for the configured model"
        )

    iface = JLens(lens, source_desc=desc)
    print(f"[load_jlens] {lens} — {desc}")
    return iface, lens_model
