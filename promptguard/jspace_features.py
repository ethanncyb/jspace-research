"""Sparse Jacobian-lens (JSpace) feature extraction for injection probing.

The "JSpace" features of one prompt are the lens readouts
``softmax(unembed(J_l @ h_t))`` at each hooked layer ``l`` and token position
``t`` — a per-position distribution over vocabulary concepts. Only the top-k
entries per position are kept, which preserves exactly the sparse, localized
signal that averaging destroys. Classifier-ready feature vectors are then
built by pooling a fixed concept vocabulary over positions (max-pool by
default).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

POOLING_MODES = ("max", "topm_mass", "mean", "last_token")

DEFAULT_SEED_WORDS = (
    "ignore",
    "instruction",
    "system",
    "override",
    "hidden",
    "malicious",
    "jailbreak",
    "bypass",
    "reveal",
    "disregard",
)


@dataclass
class SparseTrace:
    """Top-k lens readouts for one prompt.

    Attributes:
        indices: ``{layer: LongTensor[T, top_k]}`` vocabulary ids.
        values: ``{layer: FloatTensor[T, top_k]}`` lens-readout probabilities
            aligned with ``indices``. Probabilities (not logits) so that
            "concept absent" pools naturally to 0.
        seq_len: Number of token positions ``T`` (shared across layers).
    """

    indices: dict[int, torch.Tensor]
    values: dict[int, torch.Tensor]
    seq_len: int

    @property
    def layers(self) -> list[int]:
        return sorted(self.indices)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "indices": self.indices,
                "values": self.values,
                "seq_len": self.seq_len,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> SparseTrace:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(payload["indices"], payload["values"], int(payload["seq_len"]))


def trace_cache_path(cache_dir: str | Path, prompt: str) -> Path:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{digest}.pt"


class JSpaceFeatureExtractor:
    """Capture residuals and reduce them to top-k lens readouts per layer."""

    def __init__(
        self,
        model: LensModel,
        lens: JacobianLens,
        layers: Sequence[int],
        *,
        top_k: int = 64,
        max_seq_len: int = 512,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.model = model
        self.lens = lens
        self.layers = sorted(set(layers))
        missing = set(self.layers) - set(lens.source_layers)
        if missing:
            raise ValueError(
                f"layers {sorted(missing)} are not fitted in the lens; "
                f"source layers are {lens.source_layers}"
            )
        self.top_k = top_k
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def extract(self, prompt: str) -> SparseTrace:
        ids = self.model.encode(prompt, max_length=self.max_seq_len)
        with ActivationRecorder(self.model.layers, at=self.layers) as recorder:
            self.model.forward(ids)
        indices: dict[int, torch.Tensor] = {}
        values: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            hidden = recorder.activations[layer][0].float()
            transported = self.lens.transport(hidden, layer)
            probs = F.softmax(self.model.unembed(transported).float(), dim=-1)
            top = probs.topk(self.top_k, dim=-1)
            indices[layer] = top.indices.cpu()
            values[layer] = top.values.cpu()
        return SparseTrace(indices, values, int(ids.shape[1]))

    def extract_many(
        self,
        prompts: Sequence[str],
        *,
        cache_dir: str | Path | None = None,
    ) -> list[SparseTrace]:
        traces: list[SparseTrace] = []
        for prompt in prompts:
            path = trace_cache_path(cache_dir, prompt) if cache_dir else None
            if path is not None and path.exists():
                traces.append(SparseTrace.load(path))
                continue
            trace = self.extract(prompt)
            if path is not None:
                trace.save(path)
            traces.append(trace)
        return traces


def build_concept_vocab(
    traces: Sequence[SparseTrace],
    *,
    max_concepts: int = 8192,
    min_df: int = 2,
) -> list[int]:
    """Union of top-k token ids across traces, frequency-filtered and capped.

    Document frequency counts in how many traces a token appears (any layer,
    any position). Ranking by ``(-df, token_id)`` and returning the selected
    ids sorted makes the vocabulary deterministic.
    """

    if max_concepts <= 0:
        raise ValueError("max_concepts must be positive")
    if min_df < 1:
        raise ValueError("min_df must be at least 1")
    document_frequency: dict[int, int] = {}
    for trace in traces:
        present: set[int] = set()
        for layer in trace.layers:
            present.update(trace.indices[layer].reshape(-1).tolist())
        for token_id in present:
            document_frequency[token_id] = document_frequency.get(token_id, 0) + 1
    eligible = [
        token_id
        for token_id, count in document_frequency.items()
        if count >= min_df
    ]
    eligible.sort(key=lambda token_id: (-document_frequency[token_id], token_id))
    return sorted(eligible[:max_concepts])


def featurize(
    trace: SparseTrace,
    concept_vocab: Sequence[int],
    *,
    pooling: str = "max",
    topm: int = 3,
) -> dict[int, torch.Tensor]:
    """Pool one trace over positions into per-layer concept feature vectors.

    Returns ``{layer: Tensor[n_concepts]}``. ``pooling`` is one of
    :data:`POOLING_MODES`; ``max`` preserves "fires once anywhere" signals and
    is the default, ``topm_mass`` sums the top ``topm`` position values,
    ``mean`` and ``last_token`` are ablation baselines.
    """

    if pooling not in POOLING_MODES:
        raise ValueError(f"unknown pooling mode: {pooling}")
    if not concept_vocab:
        raise ValueError("concept_vocab must be non-empty")
    n_concepts = len(concept_vocab)
    vocab_list = list(concept_vocab)
    if vocab_list != sorted(vocab_list):
        raise ValueError("concept_vocab must be sorted ascending")
    trace_max = max(
        int(trace.indices[layer].max()) for layer in trace.layers
    )
    lookup = torch.zeros(max(vocab_list[-1], trace_max) + 1, dtype=torch.long)
    for index, token_id in enumerate(vocab_list):
        lookup[token_id] = index + 1  # 0 stays reserved for "not in vocab"

    features: dict[int, torch.Tensor] = {}
    for layer in trace.layers:
        mapped = lookup[trace.indices[layer]]  # [T, k]
        vals = trace.values[layer] * (mapped > 0)
        dense = torch.zeros(trace.seq_len, n_concepts + 1)
        # Column 0 is the drop column: out-of-vocab ids scatter their zeroed
        # values there; valid ids are unique within a row (top-k property).
        dense.scatter_(1, mapped, vals)
        dense = dense[:, 1:]
        if pooling == "max":
            pooled = dense.amax(dim=0)
        elif pooling == "topm_mass":
            pooled = dense.topk(min(topm, trace.seq_len), dim=0).values.sum(dim=0)
        elif pooling == "mean":
            pooled = dense.mean(dim=0)
        else:  # last_token
            pooled = dense[-1]
        features[layer] = pooled
    return features


class ConceptClusters:
    """K-means clusters over the unembedding rows of the concept vocabulary."""

    def __init__(self, assignments: torch.Tensor, n_clusters: int) -> None:
        if assignments.ndim != 1:
            raise ValueError("assignments must be a 1-D tensor")
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if assignments.numel() and int(assignments.max()) >= n_clusters:
            raise ValueError("assignments reference a missing cluster")
        self.assignments = assignments.long()
        self.n_clusters = n_clusters

    @classmethod
    def from_unembedding(
        cls,
        model: LensModel,
        concept_vocab: Sequence[int],
        *,
        n_clusters: int = 64,
        iterations: int = 25,
        seed: int = 7,
    ) -> ConceptClusters:
        """Cluster concept tokens by cosine similarity of their ``W_U`` rows."""

        if not concept_vocab:
            raise ValueError("concept_vocab must be non-empty")
        n_clusters = min(n_clusters, len(concept_vocab))
        ids = torch.tensor(sorted(concept_vocab), dtype=torch.long)
        weight = model.unembedding_weight[ids].float()
        weight = F.normalize(weight, dim=-1)
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(ids), generator=generator)
        centers = weight[order[:n_clusters]].clone()
        assignments = torch.zeros(len(ids), dtype=torch.long)
        for _ in range(iterations):
            assignments = (weight @ centers.T).argmax(dim=1)
            for cluster in range(n_clusters):
                members = weight[assignments == cluster]
                if members.numel():
                    centers[cluster] = F.normalize(members.mean(dim=0), dim=0)
        return cls(assignments, n_clusters)

    def aggregate(
        self, features: Mapping[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Max-pool concept features within each cluster, per layer."""

        return {
            layer: torch.zeros(self.n_clusters).scatter_reduce_(
                0, self.assignments, vector, reduce="amax"
            )
            for layer, vector in features.items()
        }


def seed_token_ids(
    tokenizer, words: Sequence[str] = DEFAULT_SEED_WORDS
) -> list[int]:
    """Single-token ids for each seed word (common casings/spacings)."""

    ids: set[int] = set()
    for word in words:
        for variant in (word, f" {word}", word.capitalize(), f" {word.capitalize()}"):
            encoded = tokenizer(variant, add_special_tokens=False).input_ids
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
    return sorted(ids)


def keyword_baseline_score(trace: SparseTrace, token_ids: Sequence[int]) -> float:
    """Max readout probability of any seed token anywhere (0 if never in top-k).

    This is the exact-keyword baseline the learned probe must beat.
    """

    if not token_ids:
        return 0.0
    targets = torch.tensor(sorted(token_ids), dtype=torch.long)
    best = 0.0
    for layer in trace.layers:
        hits = (trace.indices[layer].unsqueeze(-1) == targets).any(dim=-1)
        if bool(hits.any()):
            best = max(best, float(trace.values[layer][hits].max()))
    return best
