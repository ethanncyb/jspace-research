from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _validate_primary_gpu_placement(hf_model: Any) -> None:
    device_map = getattr(hf_model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        placements = {str(device).lower() for device in device_map.values()}
    else:
        placements: set[str] = set()
        parameters = getattr(hf_model, "parameters", None)
        if callable(parameters):
            placements.update(str(parameter.device).lower() for parameter in parameters())
        buffers = getattr(hf_model, "buffers", None)
        if callable(buffers):
            placements.update(str(buffer.device).lower() for buffer in buffers())
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

    @property
    def context_length(self) -> int:
        config = getattr(self._hf_model, "config", None)
        text_config = getattr(config, "text_config", None)
        value = getattr(text_config or config, "max_position_embeddings", None)
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError("Could not determine the pinned model context length")
        return value

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
        dictionary: torch.Tensor | None = None,
        sparsity_k: int = 25,
        screen_candidates: int = 512,
        output_window: int = 1,
        alpha: float = 0.0,
        intervention_stats: dict[str, int] | None = None,
    ) -> torch.Tensor:
        """Edit the first W decoded token states; prefill is never edited.

        Token 1 is sampled from untouched prefill. Its edited state influences
        token 2. EOS and the final sampled token need not receive a forward pass.
        """
        import math

        from .phase1.jspace import screened_nonnegative_pursuit

        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Intervention generation expects one nonempty unpadded prompt")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if (layer is None) != (dictionary is None):
            raise ValueError("Layer and J-space dictionary must be provided together")
        if layer is not None and not 0 <= layer < self.number_layers:
            raise ValueError(f"Intervention layer is out of range: {layer}")
        if dictionary is not None:
            if dictionary.ndim != 2 or dictionary.shape[1] != self.hidden_width:
                raise ValueError("Dictionary shape does not match model width")
            if type(output_window) is not int or output_window <= 0:
                raise ValueError("output_window must be a positive integer")
            if not 0 < sparsity_k <= screen_candidates:
                raise ValueError("Require 0 < sparsity_k <= screen_candidates")

        prompt_length = int(input_ids.shape[-1])
        input_ids = input_ids.to(self.input_device)
        hook_handle = None
        prefill_seen = False
        processed = 0
        edited = 0

        if layer is not None:

            def subtract_jspace(module: Any, inputs: Any, output: Any) -> Any:
                nonlocal prefill_seen, processed, edited
                tensor = output if torch.is_tensor(output) else output[0]
                if (
                    tensor.ndim != 3
                    or tensor.shape[0] != 1
                    or tensor.shape[-1] != self.hidden_width
                ):
                    raise RuntimeError("Selected residual block must return [1, sequence, hidden]")
                if not prefill_seen:
                    prefill_seen = True
                    if tensor.shape[1] != prompt_length:
                        raise RuntimeError("Unexpected prefill length at intervention layer")
                    return output
                if tensor.shape[1] != 1:
                    raise RuntimeError("Output intervention requires cached single-token decoding")
                if processed >= output_window:
                    return output
                processed += 1
                if alpha == 0.0:
                    return output
                hidden = tensor[0, -1, :]
                reconstructed, _, _ = screened_nonnegative_pursuit(
                    hidden.unsqueeze(0),
                    dictionary,
                    sparsity_k=sparsity_k,
                    screen_candidates=screen_candidates,
                )
                vector = reconstructed[0].to(device=tensor.device, dtype=tensor.dtype)
                replacement = tensor.clone()
                replacement[0, -1, :] = hidden - float(alpha) * vector
                if not bool(torch.isfinite(replacement).all()):
                    raise RuntimeError("J-space intervention produced nonfinite hidden states")
                edited += 1
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
        if layer is not None and not prefill_seen:
            raise RuntimeError("The selected residual intervention hook was never called")
        if generated.ndim != 2 or generated.shape[0] != 1:
            raise RuntimeError("Model generation returned an unexpected token shape")
        if intervention_stats is not None:
            intervention_stats.update(
                processed_output_tokens=processed, edited_output_tokens=edited
            )
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
