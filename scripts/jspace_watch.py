"""Watch J-space signatures light up while the model reads injection prompts.

Replicates the global-workspace paper's readout locally: for each prompt in a
jspace dataset CSV, decode every fitted layer's J-space
(``topk(unembed(J_l @ h))``) and flag signature words (injection / fake /
ignore / ...) as they appear — during prefill, and optionally streamed live
during generation. Ends with hit-rate stats (attack vs benign) and optional
interactive HTML slice pages for the most signature-dense prompts.

Example:
    python scripts/jspace_watch.py --dataset data/experiments/jspace/dev.csv \
        --limit 32 --generate 32 --html 3 --out results/jspace_watch_dev
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jlens.hooks
import jlens.vis
from _common import LENS_FILES, load_model_and_lens
from promptguard.eval_harness import load_records
from promptguard.jspace_signatures import (
    SIGNATURE_LEXICON,
    GenerationWatch,
    SignatureHit,
    aggregate_hits,
    match_signatures,
)

DEFAULT_MODEL = "Qwen/Qwen3.5-9B-Base"
GLOSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "qwen_gloss.json.gz"


def parse_layers(spec: str | None, lens, n_layers: int, max_layers: int = 16):
    """``None`` -> all fitted layers strided to <= ``max_layers``; ``"late"``
    -> the late band; ``"a-b"`` or ``"a,b,c"`` -> explicit layers."""
    fitted = lens.source_layers
    if spec is None:
        stride = max(1, math.ceil(len(fitted) / max_layers))
        layers = fitted[::stride]
        if fitted[-1] not in layers:
            layers.append(fitted[-1])
        return layers
    if spec == "late":
        start = n_layers // 3
        stop = max(start + 1, n_layers - max(2, n_layers // 6))
        layers = [layer for layer in range(start, stop) if layer in fitted]
        if not layers:
            raise ValueError("late band has no overlap with fitted lens layers")
        return layers
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-", 1))
        layers = list(range(lo, hi + 1))
    else:
        layers = [int(x) for x in spec.split(",")]
    unknown = sorted(set(layers) - set(fitted))
    if unknown:
        raise ValueError(f"layers {unknown} not fitted; fitted: {fitted}")
    return sorted(set(layers))


def prefill_hits(
    lens, model, prompt: str, *, layers: list[int], top_k: int, positions_last: int,
    max_seq_len: int,
) -> tuple[list[SignatureHit], dict[int, list[tuple[str, float]]], torch.Tensor]:
    """Decode the prompt's J-space per layer; flag signature words.

    For each layer, takes the max lens logit over the scanned positions per
    vocab token, then the top-k of that vector. ``positions_last`` > 0 scans
    only the last N positions; ``positions_last`` == 0 scans every position
    (needed when the injection sits mid-prompt, e.g. BIPIA).
    Returns (hits, per-layer top words, input_ids).

    Unlike ``JacobianLens.apply`` this never materializes or copies full-vocab
    logits: residuals are windowed first, the unembed runs in the model's dtype
    on GPU, and only top-k ids leave the device (GB/s-scale speedup per
    prompt, which matters over an 11k-prompt benchmark).
    """
    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    with jlens.hooks.ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)

    weight_dtype = model.unembedding_weight.dtype
    n_window = seq_len if positions_last == 0 else min(positions_last, seq_len)
    pos_offset = seq_len - n_window
    hits: list[SignatureHit] = []
    top_words: dict[int, list[tuple[str, float]]] = {}
    for layer in layers:
        residual = recorder.activations[layer][0].detach()  # [seq_len, d_model]
        window = residual[-n_window:].float()  # transport needs fp32 J
        z = lens.transport(window, layer)  # [n_window, d_model]
        logits = model.unembed(z.to(weight_dtype)).float()  # [n_window, vocab]
        best_vals, best_pos = logits.max(dim=0)  # per-vocab over positions
        vals, ids = best_vals.topk(top_k)
        ids_list = ids.cpu().tolist()
        vals_list = vals.cpu().tolist()
        pos_list = best_pos[ids].cpu().tolist()
        decoded = [
            model.tokenizer.decode([t], clean_up_tokenization_spaces=False)
            for t in ids_list
        ]
        top_words[layer] = [
            (tok, round(v, 4)) for tok, v in zip(decoded, vals_list, strict=True)
        ]
        hit_pairs = set(match_signatures(decoded))
        for rank, (tok, val) in enumerate(zip(decoded, vals_list, strict=True), start=1):
            for group, matched in hit_pairs:
                if matched == tok:
                    hits.append(
                        SignatureHit(
                            group=group,
                            token=tok,
                            layer=layer,
                            position=pos_offset + pos_list[rank - 1],
                            logit=round(val, 4),
                            rank=rank,
                            phase="prefill",
                        )
                    )
    return hits, top_words, input_ids


def format_prefill_line(hits: list[SignatureHit], layers: list[int]) -> str:
    by_layer: dict[int, list[str]] = {layer: [] for layer in layers}
    for hit in hits:
        by_layer[hit.layer].append(f"{hit.group}:{hit.token.strip()}#{hit.rank}")
    parts = [f"L{layer}: {' '.join(items)}" for layer, items in by_layer.items() if items]
    return " | ".join(parts) if parts else "(no signature hits)"


def load_glosses() -> dict[int, str] | None:
    if not GLOSS_PATH.exists():
        return None
    with gzip.open(GLOSS_PATH, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): v for k, v in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="jspace CSV (baseline,prompt,label,category)")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(LENS_FILES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--layers", default=None, help="'late', 'a-b', 'a,b,c'; default: fitted layers strided")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--positions-last", type=int, default=8,
                        help="scan only the last N positions; 0 = scan every position")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--generate", type=int, default=0, help="max new tokens; 0 disables generation watch")
    parser.add_argument("--html", type=int, default=0, help="write slice pages for the K most signature-dense prompts")
    parser.add_argument("--out", default="results/jspace_watch")
    args = parser.parse_args()

    records = load_records(args.dataset)
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} records from {args.dataset}")

    hf, model, lens, tok, device = load_model_and_lens(args.model)
    layers = parse_layers(args.layers, lens, model.n_layers)
    print(f"watching layers {layers}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t0 = time.perf_counter()
    for i, record in enumerate(records):
        hits, top_words, input_ids = prefill_hits(
            lens, model, record.prompt,
            layers=layers, top_k=args.top_k,
            positions_last=args.positions_last, max_seq_len=args.max_seq_len,
        )
        # Prefill is a single forward pass: all its hits share one timestamp.
        prefill_t = time.perf_counter() - t0
        for hit in hits:
            hit.time_s = prefill_t
        generation = None
        if args.generate > 0:
            def on_record(rec, _i=i):
                if rec["hits"] and rec["call_index"] > 0:
                    fired = ", ".join(f"{h['group']}:{h['token'].strip()}" for h in rec["hits"])
                    print(
                        f"    [{_i}] t={rec['t'] - t0:7.1f}s step {rec['call_index'] - 1} "
                        f"L{rec['layer']}: {fired}"
                    )

            with GenerationWatch(model, lens, layers=layers, top_k=args.top_k, on_record=on_record) as watch:
                gen_ids = hf.generate(
                    input_ids,
                    max_new_tokens=args.generate,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            generation = tok.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            hits.extend(watch.hits_as_signature_hits(
                generated_offset=input_ids.shape[1], t0=t0
            ))

        groups = sorted({h.group for h in hits})
        print(
            f"[{i + 1}/{len(records)}] t={time.perf_counter() - t0:7.1f}s "
            f"label={record.label} {record.category} "
            f"groups={groups or '-'}\n    {format_prefill_line(hits, layers)}"
        )
        results.append(
            {
                "index": i,
                "label": record.label,
                "category": record.category,
                "prompt": record.prompt,
                "groups": groups,
                "hits": [asdict(h) for h in hits],
                "generation": generation,
            }
        )

    summary = aggregate_hits(results)
    summary["model"] = args.model
    summary["layers"] = layers
    summary["lexicon"] = SIGNATURE_LEXICON

    print("\n== signature hit rates ==")
    for label, stats in summary["by_label"].items():
        group_str = " ".join(
            f"{g}={r:.2f}" for g, r in stats["group_hit_rates"].items() if r > 0
        )
        print(f"label={label} (n={stats['n']}): any={stats['hit_rate']:.2f} {group_str}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "prompts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["index", "label", "category", "groups", "n_hits", "first_hit_layer", "first_hit_s"]
        )
        for result in results:
            timed = [h for h in result["hits"] if h.get("time_s") is not None]
            first = min(timed, key=lambda h: h["time_s"]) if timed else None
            writer.writerow([
                result["index"], result["label"], result["category"],
                "|".join(result["groups"]), len(result["hits"]),
                first["layer"] if first else "",
                round(first["time_s"], 2) if first else "",
            ])
    with (out_dir / "hits.jsonl").open("w", encoding="utf-8") as fh:
        for result in results:
            for hit in result["hits"]:
                fh.write(json.dumps({
                    "index": result["index"], "label": result["label"],
                    "category": result["category"], **hit,
                }) + "\n")
    if args.generate > 0:
        with (out_dir / "generations.jsonl").open("w", encoding="utf-8") as fh:
            for result in results:
                fh.write(json.dumps({
                    "index": result["index"], "label": result["label"],
                    "category": result["category"], "generation": result["generation"],
                    "hits": result["hits"],
                }) + "\n")

    if args.html > 0:
        glosses = load_glosses()
        slices_dir = out_dir / "slices"
        ranked = sorted(results, key=lambda r: len(r["groups"]), reverse=True)
        for result in ranked[: args.html]:
            slice_data = jlens.vis.compute_slice(
                model, lens, result["prompt"],
                top_n=10, layer_stride=max(1, math.ceil(len(lens.source_layers) / 16)),
                max_seq_len=args.max_seq_len,
            )
            page, _, _ = jlens.vis.build_page(
                slice_data, result["prompt"],
                title=f"jspace watch #{result['index']} (label={result['label']})",
                description=f"category={result['category']} groups={result['groups']}",
                alt_token=glosses,
            )
            slices_dir.mkdir(exist_ok=True)
            (slices_dir / f"prompt_{result['index']:04d}.html").write_text(page)
        print(f"wrote {min(args.html, len(ranked))} slice pages to {slices_dir}")

    print(f"wrote summary.json / prompts.csv to {out_dir}")


if __name__ == "__main__":
    main()
