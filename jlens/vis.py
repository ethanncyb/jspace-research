# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Slice visualisation for the Jacobian lens.

Produces an interactive HTML page per prompt: a position x layer heatmap of
the lens top-1 token with rank overlays and per-token rank-tracking charts.
Data ships as gzip'd typed-array bytes, either base64-embedded inline (single
self-contained file, d3 inlined too) or as ``slice.bin`` / ``meta.json`` /
``ranks/{tid}.bin`` sidecars for static hosting.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import html
import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens, SteeringResult
from jlens.protocol import LensModel

PAGE_TEMPLATE = (files("jlens") / "data" / "slice_vis.html").read_text(encoding="utf-8")

#: The page's only external dependency. The template carries a ``__D3__``
#: placeholder; :func:`_template` fills it with either the CDN tag (fetch
#: mode) or the inlined source (embed mode).
_D3_URL = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"
_D3_SRI = "sha384-CjloA8y00+1SDAUkjs099PVfnY2KmDC2BZnws9kh8D/lX1s46w6EPhpXdqMfjK6i"
_D3_TAG = (
    f'<script src="{_D3_URL}" integrity="{_D3_SRI}" crossorigin="anonymous"></script>'
)

_TEMPLATE_FOR_MODE: dict[PageMode, str] = {}


def _template(mode: PageMode) -> str:
    """Return :data:`PAGE_TEMPLATE` with ``__D3__`` filled in for ``mode``.

    ``"fetch"`` substitutes the SRI-pinned CDN tag. ``"embed"`` fetches d3
    once (verified against :data:`_D3_SRI`) and inlines it so the page has no
    network dependency; a fetch or integrity failure raises rather than
    silently emitting a page that depends on the CDN. Memoised per mode.
    """
    if mode in _TEMPLATE_FOR_MODE:
        return _TEMPLATE_FOR_MODE[mode]
    d3 = _D3_TAG
    if mode == "embed":
        import urllib.request

        try:
            with urllib.request.urlopen(_D3_URL, timeout=30) as response:
                body = response.read()
        except OSError as exc:
            raise RuntimeError(
                f"embed mode could not fetch d3 from {_D3_URL}; retry with network "
                "access or use mode='fetch'"
            ) from exc
        sri = "sha384-" + base64.b64encode(hashlib.sha384(body).digest()).decode()
        if sri != _D3_SRI:
            raise RuntimeError(f"d3 integrity check failed (got {sri})")
        d3 = f"<script>\n{body.decode()}\n</script>"
    _TEMPLATE_FOR_MODE[mode] = PAGE_TEMPLATE.replace("__D3__", d3)
    return _TEMPLATE_FOR_MODE[mode]


def notebook_iframe(page: str, *, height: int = 620):
    """Return an ``IPython.display.HTML`` showing ``page`` in a ``srcdoc``
    iframe (a real nested document, so scripts run and styles are isolated
    from the host notebook). Use with ``build_page(..., mode="embed")``."""
    import warnings

    from IPython.display import HTML

    with warnings.catch_warnings():
        # IPython suggests IFrame when it sees the tag, but IFrame has no
        # srcdoc parameter; the suggestion doesn't apply here.
        warnings.simplefilter("ignore", UserWarning)
        return HTML(
            f'<iframe style="width:100%;height:{height}px;border:0" '
            f'srcdoc="{html.escape(page)}"></iframe>'
        )


# --------------------------------------------------------------------------- #
# Token helpers
# --------------------------------------------------------------------------- #


def _ranks_of(
    logits: torch.Tensor, target_ids: torch.Tensor, *, chunk_size: int = 256
) -> torch.Tensor:
    """Full-vocab ranks of ``target_ids`` at every position, chunked over the
    sequence so peak memory is one ``[chunk_size, vocab]`` sort buffer.

    Args:
        logits: ``[seq_len, vocab]``.
        target_ids: 1-D ``[n_targets]`` (same targets at every position) or
            2-D ``[seq_len, n_targets]`` (per-position).

    Returns:
        ``[seq_len, n_targets]`` int64 ranks (0 = top).
    """
    seq_len, vocab = logits.shape
    out = torch.empty(
        seq_len, target_ids.shape[-1], dtype=torch.long, device=logits.device
    )
    arange = torch.arange(vocab, device=logits.device)
    for start in range(0, seq_len, chunk_size):
        sl = slice(start, start + chunk_size)
        sorted_idx = logits[sl].argsort(dim=-1, descending=True)
        full_rank = torch.empty_like(sorted_idx)
        full_rank.scatter_(1, sorted_idx, arange.expand_as(sorted_idx))
        idx = target_ids if target_ids.ndim == 1 else target_ids[sl]
        out[sl] = full_rank.gather(1, idx.expand(full_rank.shape[0], -1))
        del sorted_idx, full_rank
    return out


_MEANINGFUL_MASK_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _meaningful_token_mask(
    tokenizer, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """Vocab mask: ``True`` for tokens that decode to word characters. Used by
    ``mask_display=True`` to restrict which tokens are *displayed* (ranks stay
    full-vocab). Cached per tokenizer."""
    cache_key = (id(tokenizer), vocab_size)
    if cache_key in _MEANINGFUL_MASK_CACHE:
        return _MEANINGFUL_MASK_CACHE[cache_key].to(device)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except Exception:
            continue
        stripped = decoded.strip()
        if not stripped:
            continue
        if "<|" in stripped or (stripped.startswith("<") and stripped.endswith(">")):
            continue  # special tokens
        is_wordlike = all(
            ch.isalnum() or (0 < pos < len(stripped) - 1 and ch in "'-’")
            for pos, ch in enumerate(stripped)
        )
        mask[token_id] = is_wordlike
    _MEANINGFUL_MASK_CACHE[cache_key] = mask
    return mask.to(device)


# --------------------------------------------------------------------------- #
# Slice computation
# --------------------------------------------------------------------------- #


@dataclass
class SliceData:
    """Everything needed to render one slice page.

    All arrays are indexed ``[seq_len, n_layers, ...]``. ``layers`` always
    includes the model's final layer (rendered with ``J = I``, i.e. the
    model's actual output) so divergences from earlier lens rows are visible.
    ``context_token_ids``/``strs`` cover the full prompt; the slice arrays
    cover positions ``ctx_offset`` onward (``ctx_offset > 0`` only when
    ``last_n_tokens`` windowed the slice).
    """

    seq_len: int
    layers: list[int]
    context_token_ids: list[int]
    context_token_strs: list[str]
    top_ids: np.ndarray  # [seq_len, n_layers, top_n] int32
    top_ranks: np.ndarray  # [seq_len, n_layers, top_n] int32
    tracked_token_ids: list[int]
    rank_tensor: np.ndarray  # [seq_len, n_layers, n_tracked] int32
    vocab_fragment: dict[int, str]
    vocab_size: int = 0  # 0 = unknown; pages fall back to max observed rank
    #: Tokens pinned at compute time; the default pinned set for the page.
    pinned_token_ids: list[int] = field(default_factory=list)
    #: Index of the first slice position within the full prompt.
    ctx_offset: int = 0


@dataclass
class SteeringComparisonData:
    """Baseline and steered slices plus the selected target's rank change."""

    clean: SliceData
    steered: SliceData
    target_token_id: int
    target_text: str
    target_rank_delta: np.ndarray  # clean rank - steered rank; positive is better
    strength: float
    direction_mode: str


@torch.no_grad()
def compute_slice(
    model: LensModel,
    lens: JacobianLens,
    prompt: str,
    *,
    top_n: int = 10,
    max_tracked: int | None = None,
    pinned_token_ids: set[int] | None = None,
    layer_stride: int = 1,
    last_n_tokens: int | None = None,
    max_seq_len: int = 512,
    mask_display: bool = False,
) -> SliceData:
    """Compute the position x layer lens slice for ``prompt``.

    Args:
        model: The model to read out from.
        lens: A fitted :class:`~jlens.lens.JacobianLens`.
        prompt: Input text.
        top_n: Top tokens kept per ``(position, layer)`` cell.
        mask_display: Restrict displayed top-K to word-like tokens (ranks stay
            full-vocab).
        max_tracked: Cap on frequently-high-ranked tokens to keep full rank
            tensors for, in addition to ``pinned_token_ids``. ``None`` (the
            default) tracks every token that appears in any top-K cell.
        pinned_token_ids: Tokens that always get a full rank tensor.
        layer_stride: Render every Nth fitted layer.
        last_n_tokens: Compute the slice grid only for the last N positions
            (the forward pass still uses the full prompt). The page still
            shows the whole prompt and labels positions with their absolute
            indices. ``None`` (default) renders every position.
        max_seq_len: Truncate the prompt to this many tokens.
    """
    tokenizer = model.tokenizer
    pinned_token_ids = set(pinned_token_ids or ())
    final_layer = model.n_layers - 1

    if not lens.source_layers:
        raise ValueError("lens has no fitted layers (jacobians is empty)")
    fitted_layers = lens.source_layers
    layers = fitted_layers[::layer_stride]
    if fitted_layers[-1] not in layers:
        layers.append(fitted_layers[-1])
    if final_layer not in layers:
        layers.append(final_layer)
    layers = sorted(set(layers))

    input_ids = model.encode(prompt, max_length=max_seq_len)
    full_len = input_ids.shape[1]
    start = 0 if last_n_tokens is None else max(0, full_len - last_n_tokens)
    seq_len = full_len - start
    context_token_ids = input_ids[0].tolist()
    context_token_strs = [
        tokenizer.decode([t], clean_up_tokenization_spaces=False)
        for t in context_token_ids
    ]

    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        activations = {layer: recorder.activations[layer].detach() for layer in layers}

    def lens_logits(layer: int) -> torch.Tensor:
        residual = activations[layer][0, start:].float()
        if layer in lens.jacobians:
            residual = lens.transport(residual, layer)
        # else: layer == final_layer, J = I -> this row is the model's output.
        return model.unembed(residual).float().detach()  # [seq_len, vocab_size]

    n_layers = len(layers)
    top_ids = np.zeros((seq_len, n_layers, top_n), dtype=np.int32)
    top_ranks = np.zeros((seq_len, n_layers, top_n), dtype=np.int32)
    display_mask: torch.Tensor | None = None
    vocab_size = 0

    # Pass 1: per-layer top-K. Logits are not retained across layers (they
    # would dominate memory at long seq_len x large vocab x n_layers).
    for layer_idx, layer in enumerate(layers):
        logits = lens_logits(layer)
        vocab_size = int(logits.shape[-1])

        if not mask_display:
            top_idx = logits.topk(top_n, dim=-1).indices
            top_ids[:, layer_idx] = top_idx.cpu().numpy()
            top_ranks[:, layer_idx] = np.arange(top_n, dtype=np.int32)
        else:
            if display_mask is None:
                display_mask = _meaningful_token_mask(
                    tokenizer, vocab_size, logits.device
                )
            top_idx = (
                logits.masked_fill(~display_mask, float("-inf"))
                .topk(top_n, dim=-1)
                .indices
            )
            top_ids[:, layer_idx] = top_idx.cpu().numpy()
            top_ranks[:, layer_idx] = _ranks_of(logits, top_idx).cpu().numpy()
        del logits

    # Choose tracked tokens: pinned + most-frequently-high-ranked in the top-N grid.
    flat_ids = top_ids.ravel()
    flat_ranks = top_ranks.ravel()
    score_by_token: dict[int, float] = {}
    for token_id, rank in zip(flat_ids, flat_ranks, strict=True):
        score_by_token[int(token_id)] = score_by_token.get(int(token_id), 0.0) + 1.0 / (
            int(rank) + 1
        )
    by_score = sorted(score_by_token, key=score_by_token.__getitem__, reverse=True)
    tracked = sorted(set(by_score[:max_tracked]) | pinned_token_ids)

    # Pass 2: re-unembed per layer and compute tracked-token ranks chunked
    # (no full-seq argsort; peak memory is one layer's logits + a chunk sort).
    rank_tensor = np.full((seq_len, n_layers, len(tracked)), -1, dtype=np.int32)
    if tracked:
        tracked_tensor = torch.tensor(tracked, dtype=torch.long)
        for layer_idx, layer in enumerate(layers):
            logits = lens_logits(layer)
            rank_tensor[:, layer_idx] = (
                _ranks_of(logits, tracked_tensor.to(logits.device)).cpu().numpy()
            )
            del logits

    vocab_ids = (
        set(int(t) for t in np.unique(flat_ids)) | set(tracked) | set(context_token_ids)
    )
    vocab_fragment = {
        int(t): tokenizer.decode([int(t)], clean_up_tokenization_spaces=False)
        for t in vocab_ids
    }

    return SliceData(
        seq_len=seq_len,
        layers=layers,
        context_token_ids=context_token_ids,
        context_token_strs=context_token_strs,
        top_ids=top_ids,
        top_ranks=top_ranks,
        tracked_token_ids=tracked,
        rank_tensor=rank_tensor,
        vocab_fragment=vocab_fragment,
        vocab_size=vocab_size,
        pinned_token_ids=sorted(pinned_token_ids),
        ctx_offset=start,
    )


def _slice_from_recorded_activations(
    model: LensModel,
    lens: JacobianLens,
    input_ids: torch.Tensor,
    activations: dict[int, torch.Tensor],
    layers: list[int],
    *,
    top_n: int,
    max_tracked: int | None,
    pinned_token_ids: set[int],
    last_n_tokens: int | None,
    mask_display: bool,
) -> SliceData:
    """Build :class:`SliceData` from an already-recorded forward pass."""
    tokenizer = model.tokenizer
    full_len = input_ids.shape[1]
    start = 0 if last_n_tokens is None else max(0, full_len - last_n_tokens)
    seq_len = full_len - start
    context_token_ids = input_ids[0].detach().cpu().tolist()
    context_token_strs = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in context_token_ids
    ]

    def layer_logits(layer: int) -> torch.Tensor:
        residual = activations[layer][0, start:].float()
        if layer in lens.jacobians:
            residual = lens.transport(residual, layer)
        return model.unembed(residual).float().detach()

    n_layers = len(layers)
    top_ids = np.zeros((seq_len, n_layers, top_n), dtype=np.int32)
    top_ranks = np.zeros((seq_len, n_layers, top_n), dtype=np.int32)
    display_mask: torch.Tensor | None = None
    vocab_size = 0
    for layer_index, layer in enumerate(layers):
        logits = layer_logits(layer)
        vocab_size = int(logits.shape[-1])
        if mask_display:
            if display_mask is None:
                display_mask = _meaningful_token_mask(
                    tokenizer, vocab_size, logits.device
                )
            top_idx = (
                logits.masked_fill(~display_mask, float("-inf"))
                .topk(top_n, dim=-1)
                .indices
            )
            top_ids[:, layer_index] = top_idx.cpu().numpy()
            top_ranks[:, layer_index] = _ranks_of(logits, top_idx).cpu().numpy()
        else:
            top_idx = logits.topk(top_n, dim=-1).indices
            top_ids[:, layer_index] = top_idx.cpu().numpy()
            top_ranks[:, layer_index] = np.arange(top_n, dtype=np.int32)
        del logits

    flat_ids = top_ids.ravel()
    flat_ranks = top_ranks.ravel()
    score_by_token: dict[int, float] = {}
    for token_id, rank in zip(flat_ids, flat_ranks, strict=True):
        score_by_token[int(token_id)] = score_by_token.get(int(token_id), 0) + 1 / (
            int(rank) + 1
        )
    by_score = sorted(score_by_token, key=score_by_token.__getitem__, reverse=True)
    tracked = sorted(set(by_score[:max_tracked]) | pinned_token_ids)
    rank_tensor = np.full((seq_len, n_layers, len(tracked)), -1, dtype=np.int32)
    if tracked:
        tracked_tensor = torch.tensor(tracked, dtype=torch.long)
        for layer_index, layer in enumerate(layers):
            logits = layer_logits(layer)
            rank_tensor[:, layer_index] = (
                _ranks_of(logits, tracked_tensor.to(logits.device)).cpu().numpy()
            )
            del logits
    vocab_ids = (
        set(int(token_id) for token_id in np.unique(flat_ids))
        | set(tracked)
        | set(context_token_ids)
    )
    vocab_fragment = {
        token_id: tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False
        )
        for token_id in vocab_ids
    }
    return SliceData(
        seq_len=seq_len,
        layers=layers,
        context_token_ids=context_token_ids,
        context_token_strs=context_token_strs,
        top_ids=top_ids,
        top_ranks=top_ranks,
        tracked_token_ids=tracked,
        rank_tensor=rank_tensor,
        vocab_fragment=vocab_fragment,
        vocab_size=vocab_size,
        pinned_token_ids=sorted(pinned_token_ids),
        ctx_offset=start,
    )


@torch.no_grad()
def compute_steering_comparison(
    model: LensModel,
    lens: JacobianLens,
    result: SteeringResult,
    *,
    top_n: int = 10,
    max_tracked: int | None = 128,
    layer_stride: int = 1,
    last_n_tokens: int | None = None,
    mask_display: bool = False,
) -> SteeringComparisonData:
    """Convert a :class:`SteeringResult` into clean/steered slice data."""
    if layer_stride <= 0:
        raise ValueError("layer_stride must be positive")
    available = sorted(
        set(result.clean_activations) & set(result.steered_activations)
    )
    layers = available[::layer_stride]
    if available and available[-1] not in layers:
        layers.append(available[-1])
    if not layers:
        raise ValueError("steering result has no shared recorded layers")
    pinned = {result.target_token_id}
    kwargs = dict(
        top_n=top_n,
        max_tracked=max_tracked,
        pinned_token_ids=pinned,
        last_n_tokens=last_n_tokens,
        mask_display=mask_display,
    )
    clean = _slice_from_recorded_activations(
        model,
        lens,
        result.input_ids,
        result.clean_activations,
        layers,
        **kwargs,
    )
    steered = _slice_from_recorded_activations(
        model,
        lens,
        result.input_ids,
        result.steered_activations,
        layers,
        **kwargs,
    )
    clean_column = clean.tracked_token_ids.index(result.target_token_id)
    steered_column = steered.tracked_token_ids.index(result.target_token_id)
    delta = (
        clean.rank_tensor[:, :, clean_column]
        - steered.rank_tensor[:, :, steered_column]
    )
    target_text = model.tokenizer.decode(
        [result.target_token_id], clean_up_tokenization_spaces=False
    )
    return SteeringComparisonData(
        clean=clean,
        steered=steered,
        target_token_id=result.target_token_id,
        target_text=target_text,
        target_rank_delta=delta,
        strength=result.strength,
        direction_mode=result.direction_mode,
    )


# --------------------------------------------------------------------------- #
# Page rendering
# --------------------------------------------------------------------------- #

PageMode = Literal["embed", "fetch"]


def _slice_meta(
    slice_data: SliceData,
    prompt: str,
    title: str,
    description: str,
    pinned_token_ids: set[int] | None,
    alt_token: dict[int, str] | None = None,
) -> dict:
    """Everything the page needs that isn't the per-cell ``(ctx, layer, token)``
    grid: context strings, layer list, vocab fragment, tracked/pinned IDs.

    ``alt_token`` optionally maps token IDs to an alternative display string
    (e.g. an English gloss for a non-Latin token); the page renders these as
    ``token (alt)`` wherever the token appears."""
    pinned = (pinned_token_ids or set()) & set(slice_data.tracked_token_ids)
    meta = {
        "title": title,
        "what": description,
        "prompt": prompt,
        "T": slice_data.seq_len,
        "layers": slice_data.layers,
        "top_n": slice_data.top_ids.shape[2],
        "ctx_strs": slice_data.context_token_strs,
        "tracked": slice_data.tracked_token_ids,
        "vocab": {str(k): v for k, v in slice_data.vocab_fragment.items()},
        "pinned": sorted(pinned),
        **({"vocab_size": slice_data.vocab_size} if slice_data.vocab_size else {}),
        **({"ctx_offset": slice_data.ctx_offset} if slice_data.ctx_offset else {}),
    }
    if alt_token:
        meta["alt_token"] = {
            str(tid): alt_token[tid]
            for tid in slice_data.vocab_fragment
            if tid in alt_token
        }
    return meta


def _slice_bin(slice_data: SliceData) -> bytes:
    """Top-K grid as gzip'd raw little-endian bytes: ``token_id<i4 || rank<i4``,
    each block ``[seq_len, n_layers, top_n]`` row-major."""
    return gzip.compress(
        slice_data.top_ids.astype("<i4").tobytes()
        + slice_data.top_ranks.astype("<i4").tobytes(),
        compresslevel=6,
    )


def write_slice_files(
    slice_data: SliceData,
    out_dir: str | Path,
    *,
    prompt: str,
    title: str,
    description: str,
    pinned_token_ids: set[int] | None = None,
    alt_token: dict[int, str] | None = None,
) -> None:
    """Write the sidecar files a ``mode="fetch"`` page reads: ``meta.json``,
    ``slice.bin``, and one ``ranks/{token_id}.bin`` (gzip'd ``[seq_len,
    n_layers]`` int32, row-major) per tracked token. ``pinned_token_ids``
    defaults to the set recorded on ``slice_data`` by :func:`compute_slice`."""
    if pinned_token_ids is None:
        pinned_token_ids = set(slice_data.pinned_token_ids)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _slice_meta(
        slice_data, prompt, title, description, pinned_token_ids, alt_token
    )
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (out_dir / "slice.bin").write_bytes(_slice_bin(slice_data))
    ranks_dir = out_dir / "ranks"
    ranks_dir.mkdir(exist_ok=True)
    ranks = slice_data.rank_tensor.astype("<i4")
    for i, tid in enumerate(slice_data.tracked_token_ids):
        (ranks_dir / f"{tid}.bin").write_bytes(
            gzip.compress(ranks[:, :, i].tobytes(), compresslevel=6)
        )


def build_page(
    slice_data: SliceData,
    prompt: str,
    *,
    title: str,
    description: str,
    pinned_token_ids: set[int] | None = None,
    mode: PageMode = "embed",
    out_dir: str | Path | None = None,
    alt_token: dict[int, str] | None = None,
) -> tuple[str, int, int]:
    """Render ``slice_data`` into an HTML page.

    Args:
        pinned_token_ids: Tokens pinned when the page loads. Defaults to the
            set recorded on ``slice_data`` by :func:`compute_slice`.
        mode: How the page obtains its data.

            - ``"embed"`` (default): the fetch-mode files base64-embedded
              inline, and d3 inlined too, so the page is a fully
              self-contained single file (display it in a notebook with
              :func:`notebook_iframe`). Not recommended for long prompts:
              the rank files dominate the payload (roughly
              ``len(tracked) * seq_len * n_layers * 4`` bytes
              pre-compression), so use ``"fetch"`` past a few hundred
              positions.
            - ``"fetch"``: the page fetches ``./meta.json`` and
              ``./slice.bin`` at load time and ``./ranks/{tid}.bin`` lazily
              on pin; d3 loads from CDN. Requires ``out_dir``. Suited to
              static hosting; also accepts a ``?datapath=`` URL param.
        out_dir: Where to write sidecar files for ``"fetch"``.

    Returns:
        ``(html, raw_bytes, payload_bytes)``.
    """
    if pinned_token_ids is None:
        pinned_token_ids = set(slice_data.pinned_token_ids)
    meta = _slice_meta(
        slice_data, prompt, title, description, pinned_token_ids, alt_token
    )
    bootstrap: dict
    raw_bytes = payload_bytes = 0

    if mode == "embed":
        # The fetch-mode files base64-encoded inline; the loader treats this
        # as a virtual filesystem and decodes ranks lazily on pin.
        ranks = slice_data.rank_tensor.astype("<i4")
        files = {"slice.bin": _slice_bin(slice_data)} | {
            f"ranks/{tid}.bin": gzip.compress(ranks[:, :, i].tobytes(), compresslevel=6)
            for i, tid in enumerate(slice_data.tracked_token_ids)
        }
        bootstrap = {
            "mode": "embed",
            "meta": meta,
            "files": {
                name: base64.b64encode(body).decode() for name, body in files.items()
            },
        }
        raw_bytes = slice_data.top_ids.nbytes * 2 + slice_data.rank_tensor.nbytes
        payload_bytes = sum(len(b) for b in files.values())

    elif mode == "fetch":
        if out_dir is None:
            raise ValueError("mode='fetch' requires out_dir")
        write_slice_files(
            slice_data,
            out_dir,
            prompt=prompt,
            title=title,
            description=description,
            pinned_token_ids=pinned_token_ids,
            alt_token=alt_token,
        )
        bootstrap = {"mode": "fetch"}
        payload_bytes = (Path(out_dir) / "slice.bin").stat().st_size

    else:
        raise ValueError(f"unknown mode {mode!r}")

    # ``</`` -> ``<\/`` so a vocab string can't close the <script> tag.
    bootstrap_json = json.dumps(bootstrap, ensure_ascii=False).replace("</", "<\\/")
    page = (
        _template(mode)
        .replace("__TITLE__", html.escape(title))
        .replace("__WHAT__", html.escape(description))
        .replace("__BOOTSTRAP__", bootstrap_json)
    )
    return page, raw_bytes, payload_bytes


def build_steering_comparison_page(
    comparison: SteeringComparisonData,
    *,
    title: str = "JSpace steering comparison",
    description: str = "Baseline and steered Jacobian-lens readouts.",
) -> str:
    """Render clean, steered, and target-rank-delta layer × position grids."""
    clean = comparison.clean
    steered = comparison.steered
    if clean.layers != steered.layers or clean.seq_len != steered.seq_len:
        raise ValueError("clean and steered slices must have matching grids")
    clean_target_column = clean.tracked_token_ids.index(comparison.target_token_id)
    steered_target_column = steered.tracked_token_ids.index(
        comparison.target_token_id
    )

    def top_text(slice_data: SliceData) -> list[list[str]]:
        return [
            [
                slice_data.vocab_fragment.get(
                    int(slice_data.top_ids[position, layer, 0]),
                    str(int(slice_data.top_ids[position, layer, 0])),
                )
                for layer in range(len(slice_data.layers))
            ]
            for position in range(slice_data.seq_len)
        ]

    positions = [
        {
            "index": index + clean.ctx_offset,
            "token": clean.context_token_strs[index + clean.ctx_offset],
        }
        for index in range(clean.seq_len)
    ]
    payload = {
        "layers": clean.layers,
        "positions": positions,
        "target_id": comparison.target_token_id,
        "target": comparison.target_text,
        "strength": comparison.strength,
        "direction_mode": comparison.direction_mode,
        "clean_top": top_text(clean),
        "steered_top": top_text(steered),
        "clean_rank": clean.rank_tensor[:, :, clean_target_column].tolist(),
        "steered_rank": steered.rank_tensor[:, :, steered_target_column].tolist(),
        "rank_delta": comparison.target_rank_delta.tolist(),
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root {{ color-scheme:dark; font-family:ui-sans-serif,system-ui,sans-serif }}
body {{ margin:18px;background:#0d1117;color:#e6edf3 }} h1 {{ margin:0 0 6px;font-size:22px }}
.desc,.meta {{ color:#9da7b3;margin-bottom:10px }} .panels {{ display:grid;grid-template-columns:1fr;gap:18px }}
.wrap {{ overflow:auto;max-height:62vh;border:1px solid #30363d;border-radius:8px }}
table {{ border-collapse:separate;border-spacing:0;font-size:10px }} th,td {{ padding:5px 7px;min-width:68px;max-width:105px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-right:1px solid #30363d;border-bottom:1px solid #30363d }}
th {{ position:sticky;top:0;background:#161b22;z-index:2 }} th:first-child,td:first-child {{ position:sticky;left:0;background:#161b22;z-index:1;min-width:115px }} th:first-child {{ z-index:3 }}
.rank {{ display:block;color:#8b949e;font-size:9px }} .positive {{ color:#7ee787 }} .negative {{ color:#ff7b72 }}
</style></head><body><h1>{html.escape(title)}</h1><div class="desc">{html.escape(description)}</div>
<div id="meta" class="meta"></div><div id="panels" class="panels"></div>
<script id="comparison-data" type="application/json">{payload_json}</script><script>
const d=JSON.parse(document.getElementById('comparison-data').textContent);
document.getElementById('meta').textContent=`Target: ${{JSON.stringify(d.target)}} (token ${{d.target_id}}) · strength ${{d.strength}} · ${{d.direction_mode}} direction`;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function panel(name,top,ranks,delta=false) {{
 const section=document.createElement('section'); section.innerHTML=`<h2>${{name}}</h2><div class="wrap"><table></table></div>`;
 const table=section.querySelector('table'); const head=document.createElement('tr');
 head.innerHTML='<th>position / token</th>'+d.layers.map(l=>`<th>L${{l}}</th>`).join(''); table.appendChild(head);
 const maxDelta=Math.max(1,...d.rank_delta.flat().map(Math.abs));
 d.positions.forEach((p,pi)=>{{ const row=document.createElement('tr'); row.innerHTML=`<td>${{p.index}}: ${{esc(p.token)}}</td>`;
  d.layers.forEach((_,li)=>{{ const td=document.createElement('td');
   if(delta){{ const v=d.rank_delta[pi][li]; const intensity=Math.min(0.85,Math.abs(v)/maxDelta); td.style.background=v>=0?`rgba(35,134,54,${{intensity}})`:`rgba(218,54,51,${{intensity}})`; td.className=v>0?'positive':v<0?'negative':''; td.innerHTML=`${{v>0?'+':''}}${{v}}<span class="rank">${{d.clean_rank[pi][li]+1}} → ${{d.steered_rank[pi][li]+1}}</span>`; }}
   else td.innerHTML=`${{esc(top[pi][li])}}<span class="rank">target rank ${{ranks[pi][li]+1}}</span>`;
   row.appendChild(td); }}); table.appendChild(row); }}); return section;
}}
const panels=document.getElementById('panels'); panels.append(panel('Clean J-lens readout',d.clean_top,d.clean_rank)); panels.append(panel('Steered J-lens readout',d.steered_top,d.steered_rank)); panels.append(panel('Target-rank improvement (clean − steered)',null,null,true));
</script></body></html>"""
