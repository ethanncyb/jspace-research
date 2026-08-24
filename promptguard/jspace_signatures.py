"""Lexicon-based J-space signature detection.

Replicates the readout style of the global-workspace / Jacobian-lens paper:
decode each layer's J-space (``topk(unembed(J_l @ h))``) and flag curated
"signature" words — e.g. ``injection``, ``fake``, ``ignore`` — when they appear
in the decode. Two readout paths share the same matching:

* prefill: per-position lens logits from ``JacobianLens.apply`` (the model
  reading the prompt — the paper's prompt-injection case);
* generation: :class:`GenerationWatch`, observation-only forward hooks that
  decode the last position on every ``generate`` step.

Aggregation (:func:`aggregate_hits`) turns per-prompt hits into hit-rate stats
per label / category / signature group. Everything here except the hook bodies
is model-free and unit-tested in ``tests/test_jspace_signatures.py``.
"""

from __future__ import annotations

import string
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import torch

# Signature groups: word fragments matched case-insensitively against decoded
# top-k token strings. Substring matching, so "inject" covers
# injection/injected/injecting.
SIGNATURE_LEXICON: dict[str, tuple[str, ...]] = {
    "injection": ("inject",),
    "deception": ("fake", "forged", "forgery", "spoof", "impersonat", "decei"),
    "override": ("ignore", "disregard", "override", "forget", "bypass"),
    "instruction": ("instruction", "system", "prompt", "directive", "command"),
    "jailbreak": ("jailbreak", "dan", "unrestricted", "uncensored"),
    # BIPIA-style indirect injections are phrased "in your response, ..." —
    # the J-space decode at the injection span lights up with the model
    # planning a reply to the embedded instruction (incl. 回复 for Qwen).
    "response": ("response", "reply", "回复"),
}

_PUNCT = string.punctuation


def normalize_token(token: str) -> str:
    """Normalize a decoded token string for lexicon matching.

    Lowercases and strips surrounding whitespace, punctuation, and BPE word
    markers (``Ġ``/``▁``) so " injection", "ĠInjection!", "FAKE" all match.
    Markers are stripped before lowercasing (``"Ġ".lower()`` is ``"ġ"``).
    """
    return (
        token.replace("Ġ", " ").replace("▁", " ").strip().lower().strip(_PUNCT).strip()
    )


# Substring collisions: decoded tokens that contain a needle but are
# semantically unrelated ("unforgettable" contains "forget"/"forg").
_COLLISIONS: dict[str, tuple[str, ...]] = {
    "forgettable": ("forget",),
    "unforgettable": ("forget",),
}


def match_signatures(
    token_strs: Iterable[str],
    lexicon: dict[str, tuple[str, ...]] | None = None,
) -> list[tuple[str, str]]:
    """Return deduplicated ``(group, token)`` hits for decoded token strings."""
    lexicon = SIGNATURE_LEXICON if lexicon is None else lexicon
    hits: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in token_strs:
        norm = normalize_token(token)
        if not norm:
            continue
        blocked = _COLLISIONS.get(norm, ())
        for group, needles in lexicon.items():
            # Short needles (<=3 chars, e.g. "dan") match exactly to avoid
            # firing on "jordan"/"dance"; longer ones match as substrings.
            if any(
                needle not in blocked
                and (needle == norm if len(needle) <= 3 else needle in norm)
                for needle in needles
            ):
                key = (group, norm)
                if key not in seen:
                    seen.add(key)
                    hits.append((group, token))
    return hits


@dataclass
class SignatureHit:
    """One signature word observed in one layer's J-space decode."""

    group: str
    token: str
    layer: int
    position: int  # token position (prefill) or generation step
    logit: float
    rank: int  # 1-based rank within the top-k decode
    phase: str = "prefill"  # "prefill" | "generate"
    time_s: float | None = None  # seconds since the watch run started


def aggregate_hits(prompt_results: Iterable[dict]) -> dict:
    """Summarize per-prompt signature hits into hit-rate stats.

    Each prompt result needs ``label`` (int), ``category`` (str), and ``hits``
    (list of :class:`SignatureHit` or equivalent dicts). Returns per-label,
    per-category, and per-group breakdowns, split by phase.
    """
    results = list(prompt_results)
    by_label: dict[str, dict] = {}
    by_category: dict[str, dict] = {}
    groups = sorted(SIGNATURE_LEXICON)

    def bucket(table: dict, key: str) -> dict:
        return table.setdefault(
            key,
            {
                "n": 0,
                "n_with_hits": 0,
                "group_hits": {g: 0 for g in groups},
                "phases": {"prefill": 0, "generate": 0},
            },
        )

    def hit_group(hit) -> str:
        return hit.group if isinstance(hit, SignatureHit) else hit["group"]

    def hit_phase(hit) -> str:
        return hit.phase if isinstance(hit, SignatureHit) else hit.get("phase", "prefill")

    for result in results:
        hits = list(result.get("hits", []))
        for table, key in (
            (by_label, str(int(result["label"]))),
            (by_category, str(result["category"])),
        ):
            entry = bucket(table, key)
            entry["n"] += 1
            if hits:
                entry["n_with_hits"] += 1
                for group in {hit_group(h) for h in hits}:
                    entry["group_hits"][group] += 1
                for phase in {hit_phase(h) for h in hits}:
                    entry["phases"][phase] = entry["phases"].get(phase, 0) + 1

    def finalize(table: dict) -> dict:
        out = {}
        for key, entry in sorted(table.items()):
            n = max(entry["n"], 1)
            out[key] = {
                "n": entry["n"],
                "hit_rate": entry["n_with_hits"] / n,
                "group_hit_rates": {
                    g: entry["group_hits"][g] / n for g in groups
                },
                "phase_hit_rates": {
                    p: c / n for p, c in entry["phases"].items()
                },
            }
        return out

    return {
        "n_prompts": len(results),
        "by_label": finalize(by_label),
        "by_category": finalize(by_category),
    }


class GenerationWatch:
    """Observation-only J-space decode on every ``generate`` forward call.

    Registers forward hooks on ``model.layers[layer]``; each hook reads the
    last position's residual, transports it through the lens
    (``topk(unembed(J_l @ h))``), and records top words plus lexicon hits.
    Hooks never modify activations. ``call_index`` 0 is the prefill call;
    call ``c >= 1`` observed generated token ``c - 1``.
    """

    def __init__(
        self,
        model,
        lens,
        *,
        layers: list[int],
        top_k: int = 16,
        lexicon: dict[str, tuple[str, ...]] | None = None,
        on_record: Callable[[dict], None] | None = None,
    ) -> None:
        self._model = model
        self._lens = lens
        self._tokenizer = model.tokenizer
        self.layers = sorted(layers)
        self.top_k = int(top_k)
        self._lexicon = lexicon
        self._on_record = on_record
        self._call_index = 0
        self._handles: list = []
        self.records: list[dict] = []
        unknown = sorted(set(self.layers) - set(lens.source_layers))
        if unknown:
            raise ValueError(
                f"layers {unknown} not fitted in the lens "
                f"(fitted: {lens.source_layers})"
            )

    def _make_hook(self, layer: int):
        def hook(module, inputs, output) -> None:
            fired_at = time.perf_counter()
            hidden = output if torch.is_tensor(output) else output[0]
            # hidden: [batch, seq_len, d_model]; observe the last position only.
            last = hidden[0, -1].detach().float()
            with torch.no_grad():
                logits = self._model.unembed(self._lens.transport(last, layer))
            values, ids = logits.float().topk(self.top_k)
            top_words = [
                (
                    self._tokenizer.decode(
                        [int(t)], clean_up_tokenization_spaces=False
                    ),
                    round(float(v), 4),
                )
                for v, t in zip(values, ids, strict=True)
            ]
            hits = match_signatures((w for w, _ in top_words), self._lexicon)
            record = {
                "layer": layer,
                "call_index": self._call_index,
                "t": fired_at,
                "top_words": top_words,
                "hits": [
                    {"group": g, "token": t, "rank": [w for w, _ in top_words].index(t) + 1}
                    for g, t in hits
                ],
            }
            self.records.append(record)
            if self._on_record is not None:
                self._on_record(record)
            # Hooks fire in layer order within one forward pass; advance the
            # call index on the last hooked layer.
            if layer == self.layers[-1]:
                self._call_index += 1

        return hook

    def __enter__(self) -> "GenerationWatch":
        for layer in self.layers:
            self._handles.append(
                self._model.layers[layer].register_forward_hook(self._make_hook(layer))
            )
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def hits_as_signature_hits(
        self, generated_offset: int = 0, t0: float | None = None
    ) -> list[SignatureHit]:
        """Convert accumulated records into :class:`SignatureHit` rows.

        ``call_index`` 0 is the prefill call (phase ``prefill``); later calls
        are generation steps (phase ``generate``, position = step number).
        ``t0`` is a ``time.perf_counter()`` reference; when given, each hit's
        ``time_s`` is seconds since ``t0``.
        """
        hits: list[SignatureHit] = []
        for record in self.records:
            call = record["call_index"]
            phase = "prefill" if call == 0 else "generate"
            position = 0 if call == 0 else generated_offset + call - 1
            time_s = None if t0 is None else record["t"] - t0
            for hit in record["hits"]:
                hits.append(
                    SignatureHit(
                        group=hit["group"],
                        token=hit["token"],
                        layer=record["layer"],
                        position=position,
                        logit=float(record["top_words"][hit["rank"] - 1][1]),
                        rank=hit["rank"],
                        phase=phase,
                        time_s=time_s,
                    )
                )
        return hits
