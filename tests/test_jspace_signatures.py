"""Model-free tests for promptguard.jspace_signatures (plus a TinyDecoder
hook-shape test for GenerationWatch)."""

from __future__ import annotations

import torch

from jlens.lens import JacobianLens
from promptguard.jspace_signatures import (
    SIGNATURE_LEXICON,
    GenerationWatch,
    SignatureHit,
    aggregate_hits,
    match_signatures,
    normalize_token,
)
from tests.tiny import TinyDecoder


class TestNormalizeToken:
    def test_strips_bpe_markers_and_case(self):
        assert normalize_token(" injection") == "injection"
        assert normalize_token("ĠInjection") == "injection"
        assert normalize_token("FAKE!") == "fake"
        assert normalize_token("▁Ignore,") == "ignore"

    def test_empty_after_strip(self):
        assert normalize_token("  ") == ""
        assert normalize_token("!") == ""


class TestMatchSignatures:
    def test_expected_groups_fire(self):
        hits = match_signatures([" injection", "FAKE", "ignore", "the", "cat"])
        groups = {group for group, _ in hits}
        assert groups == {"injection", "deception", "override"}

    def test_non_words_do_not_fire(self):
        assert match_signatures(["the", "cat", "sat", "123"]) == []

    def test_short_needles_match_exactly(self):
        # "dan" must not fire on substrings of longer words.
        assert match_signatures(["Jordan", "dancing"]) == []
        assert match_signatures(["DAN"])[0][0] == "jailbreak"

    def test_substring_variants(self):
        groups = {g for g, _ in match_signatures(["injected", "disregarded", "jailbreaking"])}
        assert groups == {"injection", "override", "jailbreak"}

    def test_dedupes_group_token_pairs(self):
        hits = match_signatures(["fake", "fake", " fake "])
        assert hits == [("deception", "fake")]

    def test_custom_lexicon(self):
        hits = match_signatures(["hello"], lexicon={"greeting": ("hell",)})
        assert hits == [("greeting", "hello")]


class TestAggregateHits:
    def _result(self, label, category, hits):
        return {"label": label, "category": category, "hits": hits}

    def test_hit_rates_by_label(self):
        results = [
            self._result(1, "attack:a", [SignatureHit("injection", "injection", 10, 3, 5.0, 1)]),
            self._result(1, "attack:b", []),
            self._result(0, "benign", []),
            self._result(0, "benign", [SignatureHit("deception", "fake", 12, 0, 4.0, 2)]),
        ]
        summary = aggregate_hits(results)
        assert summary["n_prompts"] == 4
        assert summary["by_label"]["1"]["hit_rate"] == 0.5
        assert summary["by_label"]["1"]["group_hit_rates"]["injection"] == 0.5
        assert summary["by_label"]["0"]["hit_rate"] == 0.5
        assert summary["by_label"]["0"]["group_hit_rates"]["deception"] == 0.5
        # Groups that never fired still appear, at rate 0.
        assert summary["by_label"]["1"]["group_hit_rates"]["jailbreak"] == 0.0

    def test_by_category(self):
        results = [
            self._result(1, "attack:a", [SignatureHit("override", "ignore", 5, 1, 3.0, 1)]),
            self._result(1, "attack:a", []),
        ]
        summary = aggregate_hits(results)
        assert summary["by_category"]["attack:a"]["hit_rate"] == 0.5

    def test_phase_split(self):
        results = [
            self._result(
                1, "attack:a",
                [
                    SignatureHit("injection", "injection", 10, 3, 5.0, 1, phase="prefill"),
                    SignatureHit("deception", "fake", 12, 0, 4.0, 2, phase="generate"),
                ],
            )
        ]
        phases = aggregate_hits(results)["by_label"]["1"]["phase_hit_rates"]
        assert phases == {"prefill": 1.0, "generate": 1.0}

    def test_empty_input(self):
        summary = aggregate_hits([])
        assert summary["n_prompts"] == 0
        assert summary["by_label"] == {}

    def test_accepts_plain_dict_hits(self):
        results = [
            self._result(1, "attack:a", [{"group": "injection", "phase": "prefill"}])
        ]
        assert aggregate_hits(results)["by_label"]["1"]["hit_rate"] == 1.0


class TestGenerationWatch:
    def _lens(self, d_model: int, layers: list[int]) -> JacobianLens:
        return JacobianLens(
            {layer: torch.eye(d_model) for layer in layers},
            n_prompts=1,
            d_model=d_model,
        )

    def test_observation_only_and_record_shape(self):
        model = TinyDecoder(n_layers=4, d_model=8, vocab_size=32)
        lens = self._lens(8, [1, 3])
        input_ids = model.encode("hello")

        clean = model.forward(input_ids).last_hidden_state
        with GenerationWatch(model, lens, layers=[1, 3], top_k=4) as watch:
            observed = model.forward(input_ids).last_hidden_state
            model.forward(input_ids)  # second call advances call_index
        assert torch.equal(clean, observed)

        assert len(watch.records) == 4  # 2 layers x 2 calls
        record = watch.records[0]
        assert record["layer"] == 1
        assert record["call_index"] == 0
        assert len(record["top_words"]) == 4
        assert isinstance(record["hits"], list)
        # Second call records carry call_index 1.
        assert watch.records[2]["call_index"] == 1
        # Hooks are removed on exit.
        assert not any(m._forward_hooks for m in model.layers)

    def test_rejects_unfitted_layers(self):
        model = TinyDecoder(n_layers=4, d_model=8, vocab_size=32)
        lens = self._lens(8, [1])
        try:
            GenerationWatch(model, lens, layers=[2])
        except ValueError:
            return
        raise AssertionError("expected ValueError for unfitted layer")
