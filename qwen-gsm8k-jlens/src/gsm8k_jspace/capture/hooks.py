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


class JSpaceCapture:
    """Observation-only forward hooks. Hooks never return a modified output."""

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
