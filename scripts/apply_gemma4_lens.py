#!/usr/bin/env python3
"""Apply a fitted Jacobian lens to Gemma 4 12B QAT and print the per-layer readout.

    colab --auth=adc upload -s lens gemma4_12b_qat_lens.pt /content/lens.pt
    LENS_ARGS="--lens /content/lens.pt --prompt 'Fact: the capital of France is'" \
        colab --auth=adc exec -s lens --timeout 1200 -f scripts/apply_gemma4_lens.py

Shows, for each fitted layer, what the residual stream at a given position is
"disposed to say" -- the top-k vocabulary tokens after transporting the residual
into the final-layer basis with J_l and decoding through the model's unembedding.

Self-contained: `colab exec -f` ships a single file, so this cannot import a
shared helper. See fit_gemma4_lens.py for why the warm-up avoids inference_mode.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

MODEL_ID = "google/gemma-4-12B-it-qat-w4a16-ct"

BOOTSTRAP = {
    "compressed_tensors": "compressed-tensors",
    "hf_transfer": "hf_transfer",
    "jlens": "git+https://github.com/anthropics/jacobian-lens.git",
}


def bootstrap() -> None:
    import importlib.util

    missing = [pkg for mod, pkg in BOOTSTRAP.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"[lens] installing {' '.join(missing)}", file=sys.stderr, flush=True)
    for installer in (["uv", "pip", "install", "--system", "-q"],
                      [sys.executable, "-m", "pip", "install", "-q"]):
        if subprocess.run(installer + missing).returncode == 0:
            return
    raise SystemExit("[lens] dependency install failed")


def script_argv() -> list[str]:
    """See fit_gemma4_lens.script_argv -- `colab exec -f` hides our real argv."""
    import shlex

    if os.environ.get("LENS_ARGS"):
        return shlex.split(os.environ["LENS_ARGS"])
    argv = sys.argv[1:]
    in_kernel = "colab_kernel_launcher" in sys.argv[0] or (
        "-f" in argv and any(a.endswith(".json") for a in argv))
    return [] if in_kernel else argv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", default="/content/gemma4_12b_qat_lens.pt")
    ap.add_argument("--prompt",
                    default="Fact: The currency used in the country shaped like a boot is")
    ap.add_argument("--position", type=int, default=-1,
                    help="token position to read out (negative counts from the end)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--layers", default=None,
                    help="comma-separated subset of the fitted layers")
    ap.add_argument("--logit-lens", action="store_true",
                    help="also show the no-Jacobian baseline for comparison")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args(script_argv())

    bootstrap()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import transformers
    import jlens

    processor = transformers.AutoProcessor.from_pretrained(args.model)
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto")
    model.eval()
    with torch.no_grad():  # never inference_mode -- see fit_gemma4_lens.py
        model(input_ids=torch.tensor([[2, 108, 109]], device=model.device),
              use_cache=False)

    lens_model = jlens.from_hf(model, processor.tokenizer)
    lens = jlens.JacobianLens.load(args.lens)
    print(f"[lens] {lens_model}")
    print(f"[lens] {lens}")

    layers = ([int(x) for x in args.layers.split(",")]
              if args.layers else list(lens.source_layers))

    tok = processor.tokenizer

    def show(title: str, use_jacobian: bool) -> None:
        lens_logits, model_logits, input_ids = lens.apply(
            lens_model, args.prompt, layers=layers,
            positions=[args.position], use_jacobian=use_jacobian)
        tokens = [tok.decode([t]) for t in input_ids[0].tolist()]
        print(f"\n=== {title} ===")
        print(f"prompt   : {args.prompt!r}")
        print(f"position : {args.position} -> {tokens[args.position]!r} "
              f"(seq_len {len(tokens)})")
        for layer in layers:
            probs = lens_logits[layer][0].softmax(-1)
            top = probs.topk(args.top_k)
            preview = "  ".join(
                f"{tok.decode([i])!r}:{p:.2f}"
                for p, i in zip(top.values.tolist(), top.indices.tolist()))
            print(f"  L{layer:<3} {preview}")
        final = model_logits[0].softmax(-1).topk(args.top_k)
        print("  " + "-" * 60)
        print("  model " + "  ".join(
            f"{tok.decode([i])!r}:{p:.2f}"
            for p, i in zip(final.values.tolist(), final.indices.tolist())))

    show("jacobian lens", use_jacobian=True)
    if args.logit_lens:
        show("logit lens baseline (no J transport)", use_jacobian=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
