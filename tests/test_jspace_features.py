"""Unit tests for promptguard.jspace_features (no model required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from promptguard.jspace_features import (
    ConceptClusters,
    JSpaceFeatureExtractor,
    SparseTrace,
    build_concept_vocab,
    featurize,
    keyword_baseline_score,
    seed_token_ids,
)


def make_trace(
    indices: dict[int, list[list[int]]], values: dict[int, list[list[float]]]
) -> SparseTrace:
    tensor_indices = {layer: torch.tensor(rows) for layer, rows in indices.items()}
    tensor_values = {layer: torch.tensor(rows) for layer, rows in values.items()}
    seq_len = next(iter(tensor_indices.values())).shape[0]
    return SparseTrace(tensor_indices, tensor_values, seq_len)


class TestFeaturize:
    def test_max_pooling_preserves_sparse_spike(self) -> None:
        # Concept 7 fires strongly at exactly one of four positions.
        trace = make_trace(
            {5: [[1, 2], [3, 4], [7, 8], [9, 10]]},
            {5: [[0.1, 0.1], [0.1, 0.1], [0.9, 0.1], [0.1, 0.1]]},
        )
        features = featurize(trace, [7], pooling="max")
        assert features[5].shape == (1,)
        assert features[5][0].item() == pytest.approx(0.9)

    def test_mean_pooling_dilutes_sparse_spike(self) -> None:
        trace = make_trace(
            {5: [[1, 2], [3, 4], [7, 8], [9, 10]]},
            {5: [[0.1, 0.1], [0.1, 0.1], [0.9, 0.1], [0.1, 0.1]]},
        )
        pooled_max = featurize(trace, [7], pooling="max")[5][0].item()
        pooled_mean = featurize(trace, [7], pooling="mean")[5][0].item()
        assert pooled_max > pooled_mean
        assert pooled_mean == pytest.approx(0.9 / 4)

    def test_last_token_pooling_reads_only_final_position(self) -> None:
        trace = make_trace(
            {5: [[7, 2], [3, 4]]},
            {5: [[0.9, 0.1], [0.2, 0.3]]},
        )
        assert featurize(trace, [7], pooling="last_token")[5][0].item() == 0.0
        trace_flipped = make_trace(
            {5: [[3, 4], [7, 2]]},
            {5: [[0.2, 0.3], [0.9, 0.1]]},
        )
        assert featurize(trace_flipped, [7], pooling="last_token")[5][
            0
        ].item() == pytest.approx(0.9)

    def test_topm_mass_sums_top_positions(self) -> None:
        trace = make_trace(
            {5: [[7], [7], [7], [7]]},
            {5: [[0.9], [0.5], [0.3], [0.1]]},
        )
        pooled = featurize(trace, [7], pooling="topm_mass", topm=3)[5][0].item()
        assert pooled == pytest.approx(0.9 + 0.5 + 0.3)

    def test_out_of_vocab_ids_do_not_corrupt_first_concept(self) -> None:
        # Regression guard for the scatter drop-column: out-of-vocab ids all
        # map to column 0 and must not overwrite concept-vocab index 0.
        trace = make_trace(
            {5: [[7, 999], [999, 998]]},
            {5: [[0.8, 0.9], [0.9, 0.9]]},
        )
        features = featurize(trace, [7], pooling="max")
        assert features[5][0].item() == pytest.approx(0.8)

    def test_unknown_pooling_rejected(self) -> None:
        trace = make_trace({5: [[1]]}, {5: [[0.5]]})
        with pytest.raises(ValueError, match="pooling"):
            featurize(trace, [1], pooling="median")

    def test_unsorted_concept_vocab_rejected(self) -> None:
        trace = make_trace({5: [[1]]}, {5: [[0.5]]})
        with pytest.raises(ValueError, match="sorted"):
            featurize(trace, [3, 1], pooling="max")


class TestBuildConceptVocab:
    def test_deterministic_and_capped(self) -> None:
        traces = [
            make_trace({5: [[1, 2]]}, {5: [[0.5, 0.5]]}),
            make_trace({5: [[1, 3]]}, {5: [[0.5, 0.5]]}),
            make_trace({5: [[1, 4]]}, {5: [[0.5, 0.5]]}),
        ]
        vocab_a = build_concept_vocab(traces, max_concepts=2, min_df=1)
        vocab_b = build_concept_vocab(traces, max_concepts=2, min_df=1)
        assert vocab_a == vocab_b
        assert vocab_a == sorted(vocab_a)
        assert len(vocab_a) == 2
        assert 1 in vocab_a  # highest document frequency survives the cap

    def test_min_df_filters_rare_tokens(self) -> None:
        traces = [
            make_trace({5: [[1, 2]]}, {5: [[0.5, 0.5]]}),
            make_trace({5: [[1, 3]]}, {5: [[0.5, 0.5]]}),
        ]
        vocab = build_concept_vocab(traces, min_df=2)
        assert vocab == [1]


class TestSparseTraceIO:
    def test_save_load_round_trip(self, tmp_path) -> None:
        trace = make_trace(
            {5: [[1, 2], [3, 4]], 9: [[5, 6], [7, 8]]},
            {5: [[0.1, 0.2], [0.3, 0.4]], 9: [[0.5, 0.6], [0.7, 0.8]]},
        )
        path = tmp_path / "trace.pt"
        trace.save(path)
        loaded = SparseTrace.load(path)
        assert loaded.layers == [5, 9]
        assert loaded.seq_len == 2
        for layer in (5, 9):
            assert torch.equal(loaded.indices[layer], trace.indices[layer])
            assert torch.equal(loaded.values[layer], trace.values[layer])


class TestKeywordBaseline:
    def test_hit_and_miss(self) -> None:
        trace = make_trace(
            {5: [[7, 2]], 9: [[3, 7]]},
            {5: [[0.4, 0.1]], 9: [[0.1, 0.6]]},
        )
        assert keyword_baseline_score(trace, [7]) == pytest.approx(0.6)
        assert keyword_baseline_score(trace, [42]) == 0.0
        assert keyword_baseline_score(trace, []) == 0.0


class FakeTokenizer:
    def __init__(self, mapping: dict[str, list[int]]):
        self.mapping = mapping

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=self.mapping.get(text, [0, 1]))


class TestSeedTokenIds:
    def test_single_token_variants_only(self) -> None:
        tokenizer = FakeTokenizer(
            {"ignore": [11], " ignore": [11], "Ignore": [12], " Ignore": [12]}
        )
        ids = seed_token_ids(tokenizer, ["ignore", "multiword"])
        assert ids == [11, 12]  # "multiword" fell back to [0, 1] and was dropped


class TestConceptClusters:
    def test_aggregate_max_pools_within_cluster(self) -> None:
        assignments = torch.tensor([0, 0, 1])
        clusters = ConceptClusters(assignments, 2)
        features = {5: torch.tensor([0.3, 0.7, 0.2])}
        pooled = clusters.aggregate(features)
        assert pooled[5].tolist() == pytest.approx([0.7, 0.2])

    def test_invalid_assignments_rejected(self) -> None:
        with pytest.raises(ValueError, match="cluster"):
            ConceptClusters(torch.tensor([0, 5]), 2)

    def test_from_unembedding_separates_obvious_clusters(self) -> None:
        # Concepts 0-3 embed near [1, .1], concepts 4-7 near [.1, 1].
        weight = torch.tensor(
            [[1.0, 0.1]] * 4 + [[0.1, 1.0]] * 4, dtype=torch.float32
        )
        model = SimpleNamespace(unembedding_weight=weight)
        clusters = ConceptClusters.from_unembedding(
            model, list(range(8)), n_clusters=2, iterations=10, seed=3
        )
        assert clusters.assignments.shape == (8,)
        # Same-cluster concepts share an assignment; cross-cluster do not.
        assert clusters.assignments[0] == clusters.assignments[3]
        assert clusters.assignments[4] == clusters.assignments[7]
        assert clusters.assignments[0] != clusters.assignments[4]


class TestExtractorValidation:
    def test_missing_lens_layers_rejected(self) -> None:
        lens = SimpleNamespace(source_layers=[1, 2, 3])
        with pytest.raises(ValueError, match="not fitted"):
            JSpaceFeatureExtractor(None, lens, [2, 7])

    def test_top_k_must_be_positive(self) -> None:
        lens = SimpleNamespace(source_layers=[1])
        with pytest.raises(ValueError, match="top_k"):
            JSpaceFeatureExtractor(None, lens, [1], top_k=0)
