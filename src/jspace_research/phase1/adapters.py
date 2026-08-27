from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .cache import sha256_file
from .config import Phase1Config


def load_tokenizer(config: Phase1Config) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.model.id, revision=config.model.revision)


class JacobianLensAdapter:
    """Small boundary around the pinned Jacobian-lens artifact."""

    def __init__(self, lens: Any, path: Path) -> None:
        self._lens = lens
        self.path = path

    @classmethod
    def load(cls, config: Phase1Config) -> JacobianLensAdapter:
        from huggingface_hub import hf_hub_download
        from jlens import JacobianLens

        path = Path(
            hf_hub_download(
                repo_id=config.lens.repository,
                filename=config.lens.filename,
                revision=config.lens.revision,
            )
        )
        actual = sha256_file(path)
        if actual != config.lens.sha256:
            raise RuntimeError(
                f"Lens SHA-256 mismatch: expected {config.lens.sha256}, found {actual}"
            )
        return cls(JacobianLens.load(str(path)), path)

    @property
    def hidden_width(self) -> int:
        return int(self._lens.d_model)

    @property
    def source_layers(self) -> tuple[int, ...]:
        return tuple(int(layer) for layer in self._lens.source_layers)

    def jacobian(self, layer: int) -> torch.Tensor:
        if layer not in self.source_layers:
            raise ValueError(f"Layer {layer} is not fitted by the configured J-lens")
        return self._lens.jacobians[layer]


class HuggingFaceModelAdapter:
    """Phase 1 model boundary for Hugging Face models supported by jlens."""

    def __init__(self, hf_model: Any, tokenizer: Any) -> None:
        import jlens

        if bool(getattr(hf_model, "is_quantized", False)):
            raise RuntimeError("The primary experiment does not permit model quantization")
        self._hf_model = hf_model
        self._model = jlens.from_hf(hf_model, tokenizer, compile=False)

        if len(tokenizer) > self.vocabulary_size:
            raise RuntimeError(
                "Tokenizer vocabulary exceeds the model unembedding vocabulary: "
                f"{len(tokenizer)} > {self.vocabulary_size}"
            )

    @classmethod
    def load(cls, config: Phase1Config, tokenizer: Any) -> HuggingFaceModelAdapter:
        from transformers import AutoModelForCausalLM

        kwargs = {
            "revision": config.model.revision,
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(config.model.id, **kwargs)
        except Exception as causal_error:
            try:
                from transformers import AutoModelForMultimodalLM

                print(
                    "AutoModelForCausalLM failed; trying AutoModelForMultimodalLM: "
                    f"{type(causal_error).__name__}"
                )
                hf_model = AutoModelForMultimodalLM.from_pretrained(config.model.id, **kwargs)
            except Exception:
                raise causal_error from None
        hf_model.eval()
        return cls(hf_model, tokenizer)

    @property
    def hidden_width(self) -> int:
        return int(self._model.d_model)

    @property
    def number_layers(self) -> int:
        return int(self._model.n_layers)

    @property
    def input_device(self) -> torch.device:
        return torch.device(self._model.input_device)

    @property
    def vocabulary_size(self) -> int:
        return int(self._model._lm_head.weight.shape[0])

    def unembedding(self) -> torch.Tensor:
        return self._model._lm_head.weight.detach().to("cpu", dtype=torch.bfloat16).contiguous()

    def capture_final_prompt_token(
        self, input_ids: torch.Tensor, layers: Sequence[int]
    ) -> torch.Tensor:
        import jlens

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Phase 1 capture expects one unpadded prompt at a time")
        requested = [int(layer) for layer in layers]
        input_ids = input_ids.to(self.input_device)
        with (
            torch.no_grad(),
            jlens.ActivationRecorder(self._model.layers, at=requested) as recorder,
        ):
            self._model.forward(input_ids)
        if set(recorder.activations) != set(requested):
            raise RuntimeError("The model adapter did not capture every requested layer")
        return torch.stack([recorder.activations[layer][0, -1, :].detach() for layer in requested])


def validate_model_lens(
    model: HuggingFaceModelAdapter,
    lens: JacobianLensAdapter,
) -> None:
    if model.hidden_width != lens.hidden_width:
        raise RuntimeError(
            f"Model/lens width mismatch: {model.hidden_width} != {lens.hidden_width}"
        )
    if not lens.source_layers:
        raise RuntimeError("The configured J-lens contains no fitted source layers")
    if min(lens.source_layers) < 0 or max(lens.source_layers) >= model.number_layers:
        raise RuntimeError("The fitted J-lens contains an out-of-range source layer")
    validate_lens_for_layers(lens, model.hidden_width, lens.source_layers)


def validate_lens_for_layers(
    lens: JacobianLensAdapter,
    hidden_width: int,
    layers: Sequence[int],
) -> None:
    if lens.hidden_width != hidden_width:
        raise RuntimeError(
            f"Cached model/lens width mismatch: {hidden_width} != {lens.hidden_width}"
        )
    if not set(layers).issubset(lens.source_layers):
        raise RuntimeError("A cached layer is not fitted by the configured J-lens")
    for layer in layers:
        jacobian = lens.jacobian(layer)
        if tuple(jacobian.shape) != (hidden_width, hidden_width):
            raise RuntimeError(
                f"J-lens layer {layer} has incompatible shape {tuple(jacobian.shape)}"
            )
