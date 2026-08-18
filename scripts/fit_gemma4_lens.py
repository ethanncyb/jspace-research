#!/usr/bin/env python3
"""Fit a Jacobian lens (github.com/anthropics/jacobian-lens) on Gemma 4 12B QAT.

    colab --auth=adc new -s lens --gpu A100
    LENS_ARGS="--n-prompts 100" \
        colab --auth=adc exec -s lens --timeout 36000 -f scripts/fit_gemma4_lens.py
    colab --auth=adc download -s lens /content/gemma4_12b_qat_lens.pt
    colab --auth=adc stop -s lens

Options go through LENS_ARGS under `colab exec` (see :func:`script_argv`); with
`colab run` they are ordinary command-line arguments.

Self-contained on purpose: `colab exec -f` / `colab run` ship a single file to the
VM, so this cannot import a shared helper module.

Two Gemma-4-QAT-specific things this handles that a stock jlens script does not:

1. `compressed-tensors` materialises the packed int4 weights as BF16 on the first
   forward pass. If that first pass happens under `torch.inference_mode()`, the
   resulting weights are *inference tensors* and every later autograd call dies with
   "Inference tensors cannot be saved for backward". So the warm-up below runs under
   `torch.no_grad()`, and nothing here may use inference_mode.
2. Gemma 4 is multimodal, so the model is an `AutoModelForMultimodalLM` and the
   tokenizer lives at `processor.tokenizer`. jlens auto-detects the text decoder at
   `model.language_model` (48 layers, d_model 3840).

Cost, measured on an A100-SXM4-40GB: one forward + ceil(d_model/dim_batch) backward
passes per prompt. Fitting *all* source layers costs the same compute as fitting one
-- every layer's gradient falls out of the same backward -- so only host RAM and
checkpoint size grow with layer count.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time

MODEL_ID = "google/gemma-4-12B-it-qat-w4a16-ct"
D_MODEL = 3840
N_LAYERS = 48

# `colab exec` lands on a bare VM; pull what the image is missing.
BOOTSTRAP = {
    "compressed_tensors": "compressed-tensors",
    "hf_transfer": "hf_transfer",
    "datasets": "datasets",
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
    """Arguments meant for us, whichever way the Colab CLI started this file.

    `colab run` sets `sys.argv` like native python. `colab exec -f` instead
    executes the file *inside the IPython kernel*, where `sys.argv` is the
    kernel's own (`-f .../kernel-<uuid>.json`) -- passing that to argparse fails.
    Under `exec`, set options via the LENS_ARGS environment variable instead:

        LENS_ARGS="--n-prompts 100" colab --auth=adc exec -s lens --timeout 36000 \
            -f scripts/fit_gemma4_lens.py
    """
    import shlex

    if os.environ.get("LENS_ARGS"):
        return shlex.split(os.environ["LENS_ARGS"])
    argv = sys.argv[1:]
    in_kernel = "colab_kernel_launcher" in sys.argv[0] or (
        "-f" in argv and any(a.endswith(".json") for a in argv))
    return [] if in_kernel else argv


def parse_layers(spec: str) -> list[int] | None:
    """`all` -> None (every layer below target); `0,4,8` or `0-47:4` -> explicit."""
    spec = spec.strip()
    if spec == "all":
        return None
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            rng, _, step = part.partition(":")
            lo, _, hi = rng.partition("-")
            out.extend(range(int(lo), int(hi) + 1, int(step) if step else 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-prompts", type=int, default=100,
                    help="WikiText-103 prompts to average over (paper uses 1000; "
                         "quality saturates near 100)")
    ap.add_argument("--source-layers", default="all",
                    help="'all', or '0,4,8', or '0-44:4'")
    ap.add_argument("--target-layer", type=int, default=-1)
    ap.add_argument("--dim-batch", type=int, default=8,
                    help="output dims per backward pass. 8 is the safe default on a "
                         "40GB A100 when fitting all layers; 16 nearly doubles speed "
                         "but peaked at 40.2GB even on a 3-layer fit")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--out", default="/content/gemma4_12b_qat_lens.pt")
    ap.add_argument("--checkpoint", default="/content/gemma4_lens_ckpt.pt")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="a full-layer checkpoint is ~2.8GB, so write it rarely")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args(script_argv())

    bootstrap()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import transformers
    import jlens
    from jlens.examples import load_wikitext_prompts

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("jlens").setLevel(logging.INFO)

    if not torch.cuda.is_available():
        print("[lens] needs a GPU", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    print(f"[lens] {props.name}, {props.total_memory / 1e9:.0f} GB")

    t0 = time.time()
    processor = transformers.AutoProcessor.from_pretrained(args.model)
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto")
    model.eval()

    # Materialise the decompressed BF16 weights OUTSIDE inference_mode -- see the
    # module docstring. Without this the first autograd call fails outright.
    with torch.no_grad():
        model(input_ids=torch.tensor([[2, 108, 109]], device=model.device),
              use_cache=False)
    print(f"[lens] model ready in {time.time() - t0:.0f}s, "
          f"VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    lens_model = jlens.from_hf(model, processor.tokenizer)
    print(f"[lens] {lens_model}")

    prompts = load_wikitext_prompts(args.n_prompts)
    print(f"[lens] {len(prompts)} prompts loaded")

    source_layers = parse_layers(args.source_layers)
    n_passes = -(-D_MODEL // args.dim_batch)
    print(f"[lens] {n_passes} backward passes/prompt, dim_batch={args.dim_batch}")

    t0 = time.time()
    lens = jlens.fit(
        lens_model,
        prompts,
        source_layers=source_layers,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    print(f"[lens] fit done in {(time.time() - t0) / 60:.1f} min")

    lens.save(args.out)
    print(f"[lens] saved -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.0f} MB, "
          f"{len(lens.source_layers)} layers, n_prompts={lens.n_prompts})")
    print(f"[lens] peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
