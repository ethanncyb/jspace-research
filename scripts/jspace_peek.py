"""Diagnostic: compare per-layer J-space top words for clean vs injected prompts.

For a few BIPIA (clean, injected) pairs, locate the injected token span
(common prefix/suffix of the token ids) and print the top-k lens words *at
that span* per layer, side by side with the same positions in the clean
prompt — so we can see what signature the model actually produces where the
injection lives (informs the watch lexicon).

Usage:
    python scripts/jspace_peek.py [--pairs 3] [--layers 20,24,28,30] [--top-k 12]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jlens.hooks
from _common import load_model_and_lens
from jspace_watch import parse_layers


def injection_span(clean_ids: list[int], inj_ids: list[int]) -> tuple[int, int]:
    """Return (lo, hi) bounds of the injected span in ``inj_ids``."""
    lo = 0
    while lo < min(len(clean_ids), len(inj_ids)) and clean_ids[lo] == inj_ids[lo]:
        lo += 1
    hi_clean, hi_inj = len(clean_ids), len(inj_ids)
    while hi_inj > lo and hi_clean > lo and clean_ids[hi_clean - 1] == inj_ids[hi_inj - 1]:
        hi_clean -= 1
        hi_inj -= 1
    return lo, hi_inj


@torch.no_grad()
def decode_positions(model, lens, prompt, *, layers, lo, hi, top_k):
    """Top-k lens words per layer, max over positions [lo, hi)."""
    input_ids = model.encode(prompt, max_length=512)
    with jlens.hooks.ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
    weight_dtype = model.unembedding_weight.dtype
    out = {}
    for layer in layers:
        window = recorder.activations[layer][0, lo:hi].detach().float()
        logits = model.unembed(lens.transport(window, layer).to(weight_dtype)).float()
        vals, ids = logits.max(dim=0).values.topk(top_k)
        out[layer] = [
            (model.tokenizer.decode([t], clean_up_tokenization_spaces=False), round(v, 3))
            for v, t in zip(vals.cpu().tolist(), ids.cpu().tolist(), strict=True)
        ]
    return out, input_ids[0].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/experiments/bipia/bipia_full.csv")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--attack-index", type=int, default=60)
    parser.add_argument("--layers", default="20,24,26,28,30")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.dataset)))
    per_context = 76  # 1 clean + 75 attacks, from build_bipia_dataset
    _, model, lens, tok, _ = load_model_and_lens(args.model)
    layers = parse_layers(args.layers, lens, model.n_layers)

    for ctx in range(args.pairs):
        base = rows[ctx * per_context]
        inj = rows[ctx * per_context + 1 + args.attack_index]
        assert base["label"] == "0" and inj["label"] == "1"

        t0 = time.perf_counter()
        inj_ids = model.encode(inj["prompt"], max_length=512)[0].tolist()
        _, clean_ids = decode_positions(
            model, lens, base["prompt"], layers=layers[:1], lo=0, hi=1, top_k=1
        )
        lo, hi = injection_span(clean_ids, inj_ids)
        span_text = tok.decode(inj_ids[lo:hi])
        inj_words, _ = decode_positions(model, lens, inj["prompt"], layers=layers, lo=lo, hi=hi, top_k=args.top_k)
        clean_words, _ = decode_positions(model, lens, base["prompt"], layers=layers, lo=lo, hi=min(hi, lo + 8), top_k=args.top_k)

        print(f"\n===== context {ctx} | {inj['category']} | span [{lo}:{hi}] = {span_text!r:.80} ({time.perf_counter() - t0:.1f}s)")
        for layer in layers:
            iw = " ".join(w.strip() or repr(w) for w, _ in inj_words[layer][:8])
            cw = " ".join(w.strip() or repr(w) for w, _ in clean_words[layer][:8])
            print(f"  L{layer} INJ  : {iw}")
            print(f"  L{layer} CLEAN: {cw}")


if __name__ == "__main__":
    main()
