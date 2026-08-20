"""Read-only J-Space capture during generation."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from gsm8k_jspace import SCHEMA_VERSION
from gsm8k_jspace.config import CaptureSection
from gsm8k_jspace.capture.selectors import should_keep_position
from gsm8k_jspace.capture.word_spans import word_end_indices
from gsm8k_jspace.models.jlens_adapter import topk_token_rows


class JSpaceCapture:
    """Observation-only forward hooks. Hooks never return a modified output."""

    REPLAY_TOKEN_MODES = frozenset({"generated_last", "full_sequence"})

    def __init__(
        self,
        lens_model,
        jlens,
        *,
        layers: list[int],
        capture_cfg: CaptureSection,
        prompt_len: int,
        run_id: str,
        example_id: str,
        condition: str,
        phase: str = "pre_intervention",
    ) -> None:
        self._blocks = lens_model.layers
        self._lens_model = lens_model
        self._jlens = jlens
        self.layers = sorted(layers)
        self.cfg = capture_cfg
        self._prompt_len = int(prompt_len)
        self._call_index = 0
        self._handles: list = []
        self.records: list[dict[str, Any]] = []
        self.run_id = run_id
        self.example_id = example_id
        self.condition = condition
        self.phase = phase

    def _make_hook(self, layer: int):
        def hook(module, inputs, output) -> None:
            if self.cfg.tokens.mode == "generated_last":
                if layer == self.layers[-1]:
                    self._call_index += 1
                return
            hidden = output if torch.is_tensor(output) else output[0]
            is_prefill = self._call_index == 0
            seq_len = hidden.shape[1]
            if is_prefill:
                positions = [seq_len - 1]
                if self.cfg.tokens.include_prompt and self.cfg.tokens.mode != "prompt_last":
                    positions = list(range(seq_len))
                elif self.cfg.tokens.mode == "prompt_last":
                    positions = [seq_len - 1]
            else:
                positions = [hidden.shape[1] - 1]
            for local_pos in positions:
                abs_pos = (
                    local_pos if is_prefill else self._prompt_len - 1 + self._call_index
                )
                generated_position = None if is_prefill else self._call_index - 1
                if not should_keep_position(
                    self.cfg.tokens,
                    call_index=self._call_index,
                    generated_position=generated_position,
                    is_prefill=is_prefill,
                ):
                    continue
                last = hidden[0, local_pos].detach()
                z = self._jlens.project_to_jspace(last, layer)
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "example_id": self.example_id,
                    "condition": self.condition,
                    "layer": layer,
                    "phase": self.phase,
                    "forward_index": self._call_index,
                    "absolute_position": int(abs_pos),
                    "generated_position": generated_position,
                    "word_index": None,
                    "token_id": None,
                    "token_text": None,
                    "capture_event": "prefill" if is_prefill else "decode",
                    "state_token_position": int(abs_pos),
                }
                if self.cfg.fields.hidden_norm:
                    record["hidden_norm"] = torch.linalg.vector_norm(last.float()).item()
                if self.cfg.fields.jspace_norm:
                    record["jspace_norm"] = torch.linalg.vector_norm(z.float()).item()
                if self.cfg.fields.top_jspace_tokens:
                    record["top_jspace_tokens"] = self._jlens.top_jspace_tokens(
                        last, layer, self._lens_model, self.cfg.top_k_tokens
                    )
                if self.cfg.fields.intervention_delta_norm:
                    record["intervention_delta_jspace_norm"] = 0.0
                    record["intervention_delta_hidden_norm"] = 0.0
                self.records.append(record)
            if layer == self.layers[-1]:
                self._call_index += 1

        return hook

    def __enter__(self) -> "JSpaceCapture":
        if self.cfg.tokens.mode in self.REPLAY_TOKEN_MODES:
            return self
        for layer in self.layers:
            self._handles.append(self._blocks[layer].register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def attach_tokens(self, generated_ids: list[int], tokenizer) -> None:
        texts = [tokenizer.decode([token_id]) for token_id in generated_ids]
        ends = word_end_indices(texts)
        word_index_by_gen = {}
        word_i = 0
        for gen_i in range(len(generated_ids)):
            word_index_by_gen[gen_i] = word_i
            if gen_i in ends:
                word_i += 1
        keep_word_end = self.cfg.tokens.mode == "word_end"
        kept: list[dict[str, Any]] = []
        for record in self.records:
            gen_pos = record.get("generated_position")
            if gen_pos is None:
                record["token_text"] = "<prompt-last>"
                if not keep_word_end:
                    kept.append(record)
                continue
            if 0 <= gen_pos < len(generated_ids):
                record["token_id"] = int(generated_ids[gen_pos])
                record["token_text"] = texts[gen_pos]
                record["word_index"] = word_index_by_gen.get(gen_pos)
            if keep_word_end and gen_pos not in ends:
                continue
            kept.append(record)
        self.records = kept

    def capture_final_replay(self, input_ids: torch.Tensor, tokenizer) -> None:
        """Observe the last emitted token by running a non-generating forward."""
        if self.cfg.tokens.mode != "generated_last":
            return
        if input_ids.ndim != 2:
            input_ids = input_ids.unsqueeze(0)
        handles = []

        def make_hook(layer: int):
            def hook(module, inputs, output) -> None:
                hidden = output if torch.is_tensor(output) else output[0]
                last = hidden[0, -1].detach()
                z = self._jlens.project_to_jspace(last, layer)
                token_id = int(input_ids[0, -1].item())
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "example_id": self.example_id,
                    "condition": self.condition,
                    "layer": layer,
                    "phase": self.phase,
                    "forward_index": -1,
                    "absolute_position": int(input_ids.shape[1] - 1),
                    "generated_position": int(input_ids.shape[1] - 1 - self._prompt_len),
                    "word_index": None,
                    "token_id": token_id,
                    "token_text": tokenizer.decode([token_id]),
                    "capture_event": "final_replay",
                    "state_token_position": int(input_ids.shape[1] - 1),
                }
                if self.cfg.fields.hidden_norm:
                    record["hidden_norm"] = torch.linalg.vector_norm(last.float()).item()
                if self.cfg.fields.jspace_norm:
                    record["jspace_norm"] = torch.linalg.vector_norm(z.float()).item()
                if self.cfg.fields.top_jspace_tokens:
                    record["top_jspace_tokens"] = self._jlens.top_jspace_tokens(
                        last, layer, self._lens_model, self.cfg.top_k_tokens
                    )
                self.records.append(record)

            return hook

        try:
            for layer in self.layers:
                handles.append(self._blocks[layer].register_forward_hook(make_hook(layer)))
            self._lens_model.forward(input_ids)
        finally:
            for handle in handles:
                handle.remove()

    def capture_sequence_replay(self, input_ids: torch.Tensor, tokenizer) -> None:
        """Record J-lens and logit-lens readouts at every token, matching jlens.apply.

        One non-generating forward over the full ``<|im_start|>`` … ``<|im_end|>``
        sequence (prompt + generated tokens). Per-layer top-k lists are stored so
        later notebooks can inspect every layer after the run.
        """
        if input_ids is None:
            return
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        token_ids = [int(x) for x in input_ids[0].tolist()]
        if not token_ids:
            return
        texts = [tokenizer.decode([token_id]) for token_id in token_ids]
        special_ids = _special_token_ids(tokenizer)
        final_layer = len(self._blocks) - 1
        record_layers = sorted(set(self.layers) | {final_layer})
        activations: dict[int, torch.Tensor] = {}
        handles = []

        def make_store(layer: int):
            def hook(module, inputs, output) -> None:
                hidden = output if torch.is_tensor(output) else output[0]
                if hidden.ndim == 3:
                    hidden = hidden[0]
                activations[layer] = hidden.detach().float().cpu()

            return hook

        try:
            for layer in record_layers:
                handles.append(self._blocks[layer].register_forward_hook(make_store(layer)))
            self._lens_model.forward(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        k = int(self.cfg.top_k_tokens)
        seq_len = len(token_ids)
        model_top: list[list[dict[str, Any]]] | None = None
        model_argmax: list[int] | None = None
        if self.cfg.fields.top_model_tokens and final_layer in activations:
            model_top, model_argmax = _position_topk(
                self._lens_model, activations[final_layer], tokenizer, k, seq_len
            )

        for layer in self.layers:
            hidden = activations.get(layer)
            if hidden is None:
                continue
            if hidden.ndim == 1:
                hidden = hidden.unsqueeze(0)
            seq = min(int(hidden.shape[0]), seq_len)
            jspace = None
            if self.cfg.fields.jspace_norm or self.cfg.fields.top_jspace_tokens:
                jspace = self._jlens.project_to_jspace(hidden[:seq], layer)
            j_top = None
            logit_top = None
            if self.cfg.fields.top_jspace_tokens and jspace is not None:
                j_top, _ = _position_topk(self._lens_model, jspace, tokenizer, k, seq)
            if self.cfg.fields.top_logit_tokens:
                logit_top, _ = _position_topk(
                    self._lens_model, hidden[:seq], tokenizer, k, seq
                )
            for pos in range(seq):
                token_id = token_ids[pos]
                last = hidden[pos]
                z = jspace[pos] if jspace is not None else None
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "example_id": self.example_id,
                    "condition": self.condition,
                    "layer": layer,
                    "phase": self.phase,
                    "forward_index": 0,
                    "absolute_position": pos,
                    "generated_position": None if pos < self._prompt_len else pos - self._prompt_len,
                    "word_index": None,
                    "token_id": token_id,
                    "token_text": texts[pos],
                    "is_special": token_id in special_ids,
                    "segment": "prompt" if pos < self._prompt_len else "generated",
                    "capture_event": "sequence_replay",
                    "state_token_position": pos,
                }
                if self.cfg.fields.hidden_norm:
                    record["hidden_norm"] = torch.linalg.vector_norm(last.float()).item()
                if self.cfg.fields.jspace_norm and z is not None:
                    record["jspace_norm"] = torch.linalg.vector_norm(z.float()).item()
                if j_top is not None:
                    record["top_jspace_tokens"] = j_top[pos]
                if logit_top is not None:
                    record["top_logit_tokens"] = logit_top[pos]
                if model_top is not None and layer == self.layers[-1]:
                    record["top_model_tokens"] = model_top[pos]
                    record["model_argmax_token_id"] = model_argmax[pos] if model_argmax else None
                    record["model_argmax_text"] = (
                        tokenizer.decode([model_argmax[pos]]) if model_argmax else None
                    )
                if self.cfg.fields.intervention_delta_norm:
                    record["intervention_delta_jspace_norm"] = 0.0
                    record["intervention_delta_hidden_norm"] = 0.0
                self.records.append(record)

    def save(self, path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(row) + "\n" for row in self.records)
        gz_path = path if str(path).endswith(".gz") else Path(str(path) + ".gz")
        with gzip.open(gz_path, "wt") as handle:
            handle.write(payload)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        layers = sorted({row["layer"] for row in self.records})
        return {
            "example_id": self.example_id,
            "path": str(gz_path.name),
            "rows": len(self.records),
            "layers": layers,
            "token_mode": self.cfg.tokens.mode,
            "sha256": "sha256:" + digest,
        }


def _as_seq_matrix(tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.float()
    if data.ndim == 3:
        data = data[0]
    if data.ndim == 1:
        data = data.unsqueeze(0)
    return data


def _position_topk(
    lens_model,
    residual: torch.Tensor,
    tokenizer,
    k: int,
    seq_len: int,
    *,
    chunk: int = 16,
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    hidden = _as_seq_matrix(residual)
    seq = min(int(hidden.shape[0]), seq_len)
    rows: list[list[dict[str, Any]]] = []
    argmax: list[int] = []
    for start in range(0, seq, chunk):
        sl = hidden[start : start + chunk]
        logits = lens_model.unembed(sl).float()
        if logits.ndim == 3:
            logits = logits[0] if logits.shape[0] == 1 else logits.reshape(-1, logits.shape[-1])
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        for local in range(logits.shape[0]):
            rows.append(topk_token_rows(logits[local], tokenizer, k))
            argmax.append(int(logits[local].argmax().item()))
    return rows[:seq], argmax[:seq]


def _special_token_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            ids.add(value)
    extra = getattr(tokenizer, "additional_special_tokens_ids", None) or []
    for item in extra:
        if item is not None:
            ids.add(int(item))
    inner = getattr(tokenizer, "_inner", tokenizer)
    convert = getattr(inner, "convert_tokens_to_ids", None)
    if convert is not None:
        for token in ("<|im_start|>", "<|im_end|>", "<|im_sep|>"):
            try:
                token_id = convert(token)
            except Exception:
                continue
            if isinstance(token_id, int) and token_id >= 0:
                ids.add(token_id)
    return ids
