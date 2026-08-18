# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Applying a fitted Jacobian lens.

A :class:`JacobianLens` holds the per-layer ``J_l`` matrices produced by
:func:`jlens.fitting.fit`. :meth:`JacobianLens.apply` runs a forward pass and
reads out the requested layers; :meth:`JacobianLens.transport` is the bare
``J_l @ h`` for callers that already have residuals.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from jlens.hooks import ActivationRecorder, ActivationSteerer, GenerateActivationSteerer
from jlens.protocol import LensModel


@dataclass
class SteeringResult:
    """Clean and intervened outputs from :meth:`JacobianLens.steer`.

    Activations retain the full sequence at each recorded layer so callers can
    construct the same layer-by-position slices used by the regular lens
    visualizer. Logits and summary diagnostics cover only ``positions``.
    Ranks are zero-based.
    """

    input_ids: torch.Tensor
    target_token_id: int
    layers: list[int]
    positions: list[int]
    strength: float
    direction_mode: Literal["jspace", "random"]
    clean_activations: dict[int, torch.Tensor]
    steered_activations: dict[int, torch.Tensor]
    clean_norms: dict[int, torch.Tensor]
    clean_logits: torch.Tensor
    steered_logits: torch.Tensor
    clean_target_ranks: torch.Tensor
    steered_target_ranks: torch.Tensor
    clean_top_token_ids: torch.Tensor
    steered_top_token_ids: torch.Tensor
    target_logit_lift: torch.Tensor
    kl_divergence: torch.Tensor


class JacobianLens:
    """A fitted Jacobian lens: per-layer ``J_l`` matrices and the readout method.

    Attributes:
        jacobians: ``{layer_index: Tensor[d_model, d_model]}``. Each ``J_l``
            maps the residual at layer ``l`` into the final-layer basis.
        source_layers: Sorted list of fitted layer indices.
        n_prompts: Number of prompts the lens was averaged over.
        d_model: Residual-stream width.
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.jacobians = {layer: J.float() for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model

    def __repr__(self) -> str:
        return (
            f"JacobianLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"source_layers=[{self.source_layers[0]}..{self.source_layers[-1]}] "
            f"({len(self.source_layers)} layers))"
        )

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        """Save to ``path``. Jacobians are stored as ``dtype`` (default fp16:
        halves file size; entries are O(1) so the range is not a constraint
        and fp16's extra mantissa bits beat bf16 here)."""
        torch.save(
            {
                "J": {layer: J.to(dtype) for layer, J in self.jacobians.items()},
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> JacobianLens:
        """Load a lens previously written by :meth:`save`."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in checkpoint:
            raise ValueError(
                f"{path} is not a JacobianLens file "
                f"(found keys {sorted(checkpoint)!r}; a fit() checkpoint?)"
            )
        return cls(
            jacobians=checkpoint["J"],
            n_prompts=checkpoint["n_prompts"],
            d_model=checkpoint["d_model"],
        )

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "lens.pt",
        revision: str | None = None,
    ) -> JacobianLens:
        """Load a lens from a local file, a local directory, or a HuggingFace
        Hub ``repo_id``. ``filename`` is the path inside the directory or repo
        (so one Hub repo can host lenses for many models); ignored when
        ``name_or_path`` is itself a file. ``revision`` selects a Hub branch,
        tag, or commit. Deserialisation goes through :meth:`load`
        (``weights_only=True``)."""
        if os.path.isfile(name_or_path):
            return cls.load(name_or_path)
        if not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download

            name_or_path = snapshot_download(
                name_or_path, allow_patterns=[filename], revision=revision
            )
        return cls.load(os.path.join(name_or_path, filename))

    @classmethod
    def merge(cls, lenses: Sequence[JacobianLens]) -> JacobianLens:
        """Combine lenses fitted on disjoint prompt subsets into one
        (``n_prompts``-weighted mean of the inputs).

        Args:
            lenses: Lenses to merge. Must agree on ``source_layers`` and
                ``d_model``.

        Raises:
            ValueError: If ``lenses`` is empty or the inputs disagree on shape.
        """
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.source_layers != first.source_layers
                or other.d_model != first.d_model
            ):
                raise ValueError("lenses disagree on source_layers / d_model")
        n_total = sum(lens.n_prompts for lens in lenses)
        merged: dict[int, torch.Tensor] = {}
        for layer in first.source_layers:
            weighted_sum = sum(
                lens.jacobians[layer] * lens.n_prompts for lens in lenses
            )
            merged[layer] = weighted_sum / n_total
        return cls(jacobians=merged, n_prompts=n_total, d_model=first.d_model)

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a residual at ``layer`` into the final-layer basis: ``J_l @ h``.

        Args:
            residual: Tensor of shape ``[..., d_model]``.
            layer: Source layer index (must be in :attr:`source_layers`).
        """
        J_bar = self.jacobians[layer].to(residual.device)
        return residual @ J_bar.T

    @torch.no_grad()
    def direction(
        self,
        model: LensModel,
        layer: int,
        token_id: int,
    ) -> torch.Tensor:
        """Return the unit J-lens vector for ``token_id`` at ``layer``.

        With row-vector activations, lens logits are approximately
        ``h @ J_l.T @ W_U.T``.  The residual-stream direction associated with
        one vocabulary token is therefore the corresponding row of
        ``W_U @ J_l``.
        """
        if layer not in self.jacobians:
            raise ValueError(
                f"layer {layer} not in source_layers; fitted layers are "
                f"{self.source_layers}"
            )
        weight = model.unembedding_weight
        if weight.ndim != 2 or weight.shape[1] != self.d_model:
            raise ValueError(
                "unembedding_weight must have shape [vocab_size, d_model]; "
                f"got {tuple(weight.shape)}"
            )
        if not 0 <= token_id < weight.shape[0]:
            raise ValueError(
                f"token_id {token_id} out of range for vocabulary size "
                f"{weight.shape[0]}"
            )
        J_bar = self.jacobians[layer].to(weight.device, dtype=torch.float32)
        vector = weight[token_id].float() @ J_bar
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(vector).all() or not torch.isfinite(norm) or norm <= 0:
            raise ValueError(
                f"J-lens direction for token {token_id} at layer {layer} "
                "is zero or non-finite"
            )
        return vector / norm

    def _default_steering_layers(self, model: LensModel) -> list[int]:
        if model.n_layers == 32:
            preferred = range(10, 27)
        else:
            start = model.n_layers // 3
            stop = max(start + 1, model.n_layers - max(2, model.n_layers // 6))
            preferred = range(start, stop)
        layers = [layer for layer in preferred if layer in self.jacobians]
        if not layers:
            raise ValueError("no fitted layers fall in the default steering band")
        return layers

    @staticmethod
    def _normalise_positions(
        positions: Sequence[int], seq_len: int
    ) -> list[int]:
        normalised: list[int] = []
        for position in positions:
            resolved = position + seq_len if position < 0 else position
            if not 0 <= resolved < seq_len:
                raise ValueError(
                    f"position {position} out of range for sequence length {seq_len}"
                )
            normalised.append(resolved)
        if not normalised:
            raise ValueError("positions must not be empty")
        if len(set(normalised)) != len(normalised):
            raise ValueError("positions resolve to duplicate indices")
        return normalised

    def _resolve_steering_layers(
        self, model: LensModel, layers: Sequence[int] | None
    ) -> list[int]:
        if layers is None:
            layers = self._default_steering_layers(model)
        layers = list(layers)
        if not layers:
            raise ValueError("layers must not be empty")
        unknown = sorted(set(layers) - set(self.source_layers))
        out_of_range = sorted(l for l in set(layers) if not 0 <= l < model.n_layers)
        if out_of_range:
            raise ValueError(
                f"layers {out_of_range} out of range for a {model.n_layers}-layer model"
            )
        if unknown:
            raise ValueError(
                f"layers {unknown} not in source_layers; fitted layers are "
                f"{self.source_layers}"
            )
        if len(set(layers)) != len(layers):
            raise ValueError("layers must not contain duplicates")
        return sorted(layers)

    def _encode_prompt(
        self,
        model: LensModel,
        prompt: str | torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        if isinstance(prompt, str):
            input_ids = model.encode(prompt, max_length=max_seq_len)
        elif torch.is_tensor(prompt):
            input_ids = prompt
        else:
            raise TypeError("prompt must be text or an input_ids tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("steer currently requires input_ids shape [1, seq_len]")
        return input_ids

    def _last_token_deltas(
        self,
        model: LensModel,
        input_ids: torch.Tensor,
        *,
        target_token_id: int,
        layers: Sequence[int],
        strength: float,
    ) -> dict[int, torch.Tensor]:
        """J-space deltas scaled by the last prompt token's clean residual norm."""
        with ActivationRecorder(model.layers, at=list(layers)) as recorder:
            model.forward(input_ids)
        deltas: dict[int, torch.Tensor] = {}
        for layer in layers:
            clean_norm = torch.linalg.vector_norm(
                recorder.activations[layer][0, -1].float()
            )
            direction = self.direction(model, layer, target_token_id)
            deltas[layer] = float(strength) * clean_norm.to(direction.device) * direction
        return deltas

    @torch.no_grad()
    def steer(
        self,
        model: LensModel,
        prompt: str | torch.Tensor,
        *,
        target_token_id: int,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] = (-1,),
        strength: float = 0.1,
        max_seq_len: int = 512,
        direction_mode: Literal["jspace", "random"] = "jspace",
        random_seed: int = 0,
    ) -> SteeringResult:
        """Run clean and steered passes for a single target vocabulary token.

        The intervention at every selected site is
        ``strength * ||h_clean|| * unit_direction``.  ``direction_mode`` can
        be ``"random"`` for a deterministic matched-norm control.
        """
        if not isinstance(strength, (int, float)) or not torch.isfinite(
            torch.tensor(float(strength))
        ):
            raise ValueError("strength must be a finite number")
        if direction_mode not in ("jspace", "random"):
            raise ValueError("direction_mode must be 'jspace' or 'random'")
        layers = self._resolve_steering_layers(model, layers)

        input_ids = self._encode_prompt(model, prompt, max_seq_len)
        resolved_positions = self._normalise_positions(positions, input_ids.shape[1])

        final_layer = model.n_layers - 1
        record_at = sorted(set(layers) | {final_layer})
        with ActivationRecorder(model.layers, at=record_at) as clean_recorder:
            model.forward(input_ids)
        clean = {
            layer: clean_recorder.activations[layer].detach() for layer in record_at
        }
        clean_norms = {
            layer: torch.linalg.vector_norm(
                clean[layer][0, resolved_positions].float(), dim=-1
            )
            for layer in layers
        }

        directions: dict[int, torch.Tensor] = {}
        for layer in layers:
            if direction_mode == "jspace":
                directions[layer] = self.direction(model, layer, target_token_id)
            else:
                generator = torch.Generator(device="cpu").manual_seed(
                    int(random_seed) + layer * 1_000_003
                )
                vector = torch.randn(self.d_model, generator=generator)
                directions[layer] = vector / torch.linalg.vector_norm(vector)
        deltas = {
            layer: float(strength)
            * clean_norms[layer].to(directions[layer].device).unsqueeze(-1)
            * directions[layer].unsqueeze(0)
            for layer in layers
        }

        # Register the modifying hooks first. Recorder hooks then observe and
        # retain the post-intervention activations.
        with ActivationSteerer(
            model.layers, deltas=deltas, positions=resolved_positions
        ), ActivationRecorder(model.layers, at=record_at) as steered_recorder:
            model.forward(input_ids)
        steered = {
            layer: steered_recorder.activations[layer].detach()
            for layer in record_at
        }

        def output_logits(activations: dict[int, torch.Tensor]) -> torch.Tensor:
            residual = activations[final_layer][0, resolved_positions].float()
            return model.unembed(residual).float()

        clean_logits_device = output_logits(clean)
        steered_logits_device = output_logits(steered)

        def target_rank(logits: torch.Tensor) -> torch.Tensor:
            target = logits[:, target_token_id].unsqueeze(-1)
            return (logits > target).sum(dim=-1)

        clean_ranks = target_rank(clean_logits_device)
        steered_ranks = target_rank(steered_logits_device)
        target_lift = (
            steered_logits_device[:, target_token_id]
            - clean_logits_device[:, target_token_id]
        )
        kl = F.kl_div(
            F.log_softmax(steered_logits_device, dim=-1),
            F.softmax(clean_logits_device, dim=-1),
            reduction="none",
        ).sum(dim=-1)

        return SteeringResult(
            input_ids=input_ids.detach(),
            target_token_id=target_token_id,
            layers=layers,
            positions=resolved_positions,
            strength=float(strength),
            direction_mode=direction_mode,
            clean_activations=clean,
            steered_activations=steered,
            clean_norms={k: v.detach().cpu() for k, v in clean_norms.items()},
            clean_logits=clean_logits_device.detach().cpu(),
            steered_logits=steered_logits_device.detach().cpu(),
            clean_target_ranks=clean_ranks.detach().cpu(),
            steered_target_ranks=steered_ranks.detach().cpu(),
            clean_top_token_ids=clean_logits_device.argmax(dim=-1).detach().cpu(),
            steered_top_token_ids=steered_logits_device.argmax(dim=-1).detach().cpu(),
            target_logit_lift=target_lift.detach().cpu(),
            kl_divergence=kl.detach().cpu(),
        )

    @torch.no_grad()
    def steer_generate(
        self,
        hf_model: Any,
        model: LensModel,
        prompt: str | torch.Tensor,
        *,
        target_token_id: int,
        strength: float = 0.1,
        decode_mode: Literal["prompt_pass", "every_step"] = "prompt_pass",
        layers: Sequence[int] | None = None,
        max_new_tokens: int = 128,
        max_seq_len: int = 512,
    ) -> torch.Tensor:
        """Greedy continuation with J-space residual deltas applied during generate.

        ``decode_mode="prompt_pass"`` writes the delta at the last prompt token
        only; cached decode steps are left unchanged (later tokens still see the
        steered residual via the KV cache).  ``decode_mode="every_step"`` also
        writes the same last-prompt-token-scaled delta at each new token.

        Scaling matches :meth:`steer`: ``strength * ||h_clean|| * unit_direction``
        at the last prompt token.  ``prompt`` is encoded with ``model.encode`` so
        the generate prefix matches the lens wrapper.

        Returns:
            The full ``generate`` sequence, shape ``[1, prompt_len + new]``.
        """
        if not isinstance(strength, (int, float)) or not torch.isfinite(
            torch.tensor(float(strength))
        ):
            raise ValueError("strength must be a finite number")
        if decode_mode not in ("prompt_pass", "every_step"):
            raise ValueError("decode_mode must be 'prompt_pass' or 'every_step'")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        layers = self._resolve_steering_layers(model, layers)
        input_ids = self._encode_prompt(model, prompt, max_seq_len)
        prompt_len = input_ids.shape[1]
        deltas = self._last_token_deltas(
            model,
            input_ids,
            target_token_id=target_token_id,
            layers=layers,
            strength=strength,
        )

        tokenizer = getattr(model, "tokenizer", None)
        pad_token_id = None
        if tokenizer is not None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id

        with GenerateActivationSteerer(
            model.layers,
            deltas=deltas,
            prompt_len=prompt_len,
            decode_mode=decode_mode,
        ):
            return hf_model.generate(**generate_kwargs)

    @torch.no_grad()
    def apply(
        self,
        model: LensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run ``model`` on ``prompt`` and return lens logits at ``positions``.

        Args:
            model: The model to read out from.
            prompt: Input text.
            layers: Layers to read out at. Defaults to all of
                :attr:`source_layers`. Must be a subset of
                :attr:`source_layers` when ``use_jacobian`` is ``True``.
            positions: Token positions to read out (Python indexing into the
                sequence; negative indices count from the end). ``None`` returns
                every position.
            max_seq_len: Truncate the prompt to this many tokens.
            use_jacobian: If ``False``, skip the ``J_l`` transport (vanilla
                logit-lens baseline).

        Returns:
            A triple ``(lens_logits, model_logits, input_ids)``. ``lens_logits``
            maps each requested layer to a ``[n_positions, vocab_size]`` tensor;
            ``model_logits`` is the model's actual final-layer logits at the
            same positions (same shape). ``n_positions`` is ``len(positions)``,
            or the full sequence length when ``positions`` is ``None``.

        Raises:
            ValueError: If any requested layer is out of range for the model,
                or (with ``use_jacobian``) not in :attr:`source_layers`.
        """
        if layers is None:
            layers = self.source_layers
        out_of_range = sorted(l for l in set(layers) if not 0 <= l < model.n_layers)
        if out_of_range:
            raise ValueError(
                f"layers {out_of_range} out of range for a {model.n_layers}-layer model"
            )
        unknown = set(layers) - set(self.source_layers)
        if use_jacobian and unknown:
            raise ValueError(
                f"layers {sorted(unknown)} not in source_layers; "
                f"fitted layers are {self.source_layers}"
            )
        final_layer = model.n_layers - 1
        record_at = sorted(set(layers) | {final_layer})

        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward(input_ids)
            activations = {i: recorder.activations[i].detach() for i in record_at}

        def select(layer: int) -> torch.Tensor:
            """Residuals at the requested positions: ``[n_positions, d_model]``."""
            full = activations[layer][0]  # [seq_len, d_model]
            return (full if positions is None else full[list(positions)]).float()

        lens_logits: dict[int, torch.Tensor] = {}
        for layer in layers:
            residual = select(layer)
            if use_jacobian:
                residual = self.transport(residual, layer)
            lens_logits[layer] = model.unembed(residual).float().cpu()

        model_logits = model.unembed(select(final_layer)).float().cpu()
        return lens_logits, model_logits, input_ids
