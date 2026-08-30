from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _validate_primary_gpu_placement(hf_model: Any) -> None:
    device_map = getattr(hf_model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        placements = {str(device).lower() for device in device_map.values()}
    else:
        parameters = getattr(hf_model, "parameters", None)
        placements = (
            {str(parameter.device).lower() for parameter in parameters()}
            if callable(parameters)
            else set()
        )
    if not placements:
        raise RuntimeError("Could not verify model placement on the primary CUDA GPU")
    if not placements.issubset({"0", "cuda", "cuda:0"}):
        raise RuntimeError(
            "The model must fit entirely on CUDA GPU 0; CPU, disk, and multi-GPU "
            f"offload are not permitted (placements: {sorted(placements)})"
        )


def load_tokenizer(config: Any) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.model.id, revision=config.model.revision)


class HuggingFaceModelAdapter:
    """Shared model boundary for activation capture and intervention generation."""

    def __init__(self, hf_model: Any, tokenizer: Any) -> None:
        import jlens

        if bool(getattr(hf_model, "is_quantized", False)):
            raise RuntimeError("The primary experiment does not permit model quantization")
        self._hf_model = hf_model
        self._model = jlens.from_hf(hf_model, tokenizer, compile=False)
        self.tokenizer = tokenizer

        if len(tokenizer) > self.vocabulary_size:
            raise RuntimeError(
                "Tokenizer vocabulary exceeds the model unembedding vocabulary: "
                f"{len(tokenizer)} > {self.vocabulary_size}"
            )

    @classmethod
    def load(cls, config: Any, tokenizer: Any) -> HuggingFaceModelAdapter:
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("Model loading requires a CUDA GPU")
        kwargs = {
            "revision": config.model.revision,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
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
        _validate_primary_gpu_placement(hf_model)
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

    def generate_from_prompt(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        layer: int | None = None,
        reconstructed_jspace: torch.Tensor | None = None,
        alpha: float = 0.0,
    ) -> torch.Tensor:
        """Greedily generate, optionally subtracting J-space once during prefill."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Intervention generation expects one unpadded prompt at a time")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if (layer is None) != (reconstructed_jspace is None):
            raise ValueError("Layer and reconstructed J-space must be provided together")
        if layer is not None and not 0 <= layer < self.number_layers:
            raise ValueError(f"Intervention layer is out of range: {layer}")
        if reconstructed_jspace is not None and tuple(reconstructed_jspace.shape) != (
            self.hidden_width,
        ):
            raise ValueError(
                "Reconstructed J-space shape does not match model width: "
                f"{tuple(reconstructed_jspace.shape)} != ({self.hidden_width},)"
            )

        prompt_length = int(input_ids.shape[-1])
        input_ids = input_ids.to(self.input_device)
        hook_handle = None
        hook_applied = False

        if layer is not None:

            def subtract_jspace(module: Any, inputs: Any, output: Any) -> Any:
                nonlocal hook_applied
                if hook_applied:
                    return output
                tensor = output if torch.is_tensor(output) else output[0]
                if tensor.ndim != 3 or tensor.shape[0] != 1:
                    raise RuntimeError(
                        "Selected residual block did not return a [1, sequence, hidden] tensor"
                    )
                if tensor.shape[-1] != self.hidden_width:
                    raise RuntimeError("Selected residual block output width is incompatible")
                replacement = tensor.clone()
                vector = reconstructed_jspace.to(device=tensor.device, dtype=tensor.dtype)
                replacement[0, -1, :] = replacement[0, -1, :] - float(alpha) * vector
                hook_applied = True
                if torch.is_tensor(output):
                    return replacement
                if isinstance(output, tuple):
                    return (replacement, *output[1:])
                if isinstance(output, list):
                    return [replacement, *output[1:]]
                raise RuntimeError(f"Unsupported residual block output type: {type(output)}")

            hook_handle = self._model.layers[layer].register_forward_hook(subtract_jspace)

        try:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
            with torch.inference_mode():
                generated = self._hf_model.generate(
                    input_ids=input_ids,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=pad_token_id,
                )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if layer is not None and not hook_applied:
            raise RuntimeError("The selected residual intervention hook was never applied")
        if generated.ndim != 2 or generated.shape[0] != 1:
            raise RuntimeError("Model generation returned an unexpected token shape")
        return generated[0, prompt_length:].detach().to("cpu")

    def generate_with_capture(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedily generate while capturing one selected-layer prefill state."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Generation capture expects one unpadded prompt at a time")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0 <= layer < self.number_layers:
            raise ValueError(f"Capture layer is out of range: {layer}")

        prompt_length = int(input_ids.shape[-1])
        input_ids = input_ids.to(self.input_device)
        captured: torch.Tensor | None = None

        def capture_prefill(module: Any, inputs: Any, output: Any) -> Any:
            nonlocal captured
            if captured is not None:
                return output
            tensor = output if torch.is_tensor(output) else output[0]
            if tensor.ndim != 3 or tensor.shape[0] != 1:
                raise RuntimeError(
                    "Selected residual block did not return a [1, sequence, hidden] tensor"
                )
            if tensor.shape[-1] != self.hidden_width:
                raise RuntimeError("Selected residual block output width is incompatible")
            captured = tensor[0, -1, :].detach().to("cpu", dtype=torch.float32)
            return output

        hook_handle = self._model.layers[layer].register_forward_hook(capture_prefill)
        try:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
            with torch.inference_mode():
                generated = self._hf_model.generate(
                    input_ids=input_ids,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=pad_token_id,
                )
        finally:
            hook_handle.remove()

        if captured is None:
            raise RuntimeError("The selected residual capture hook was never applied")
        if generated.ndim != 2 or generated.shape[0] != 1:
            raise RuntimeError("Model generation returned an unexpected token shape")
        return generated[0, prompt_length:].detach().to("cpu"), captured
