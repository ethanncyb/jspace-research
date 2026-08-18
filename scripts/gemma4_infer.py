#!/usr/bin/env -S colab --auth=adc run --gpu A100 --timeout 2400
"""Run Gemma 4 12B QAT inference on a Colab GPU.

    colab --auth=adc run --gpu A100 --timeout 2400 scripts/gemma4_infer.py "your prompt"
    ./scripts/gemma4_infer.py "your prompt"          # same thing, via the shebang

`--timeout` is load-bearing: the Colab CLI defaults to 30 seconds and the weight
download alone exceeds that. `--auth=adc` must precede the `run` subcommand.

Needs ~30 GB of VRAM. See the notebook for why: `compressed-tensors` decompresses the
packed int4 weights to BF16 on the first forward pass, so the 10.3 GB checkpoint has a
~26 GB resident footprint and will not fit on an L4 or T4.
"""

import argparse
import os
import sys
import time

W4A16 = "google/gemma-4-12B-it-qat-w4a16-ct"
MIN_VRAM_GB = 30.0

# `colab run` provisions a bare VM with no `colab install` step, so pull our own deps.
# Colab's image already has a new enough transformers; these two are what's missing.
BOOTSTRAP = ["compressed_tensors", "hf_transfer"]


def bootstrap() -> None:
    import importlib.util
    import subprocess

    missing = [m for m in BOOTSTRAP if importlib.util.find_spec(m) is None]
    if not missing:
        return
    pkgs = [m.replace("_", "-") for m in missing]
    print(f"[gemma] installing {' '.join(pkgs)}", file=sys.stderr)
    for installer in (["uv", "pip", "install", "--system", "-q"],
                      [sys.executable, "-m", "pip", "install", "-q"]):
        if subprocess.run(installer + pkgs).returncode == 0:
            return
    raise SystemExit("[gemma] dependency install failed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", nargs="*", default=[],
                    help="user prompt (reads stdin if omitted)")
    ap.add_argument("--model", default=W4A16)
    ap.add_argument("--system", default="You are a concise research assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=None,
                    help="omit for greedy decoding")
    ap.add_argument("--thinking", action="store_true",
                    help="let the model emit its reasoning trace first")
    ap.add_argument("--image", default=None, help="image URL or local path")
    args = ap.parse_args()

    prompt = " ".join(args.prompt).strip() or sys.stdin.read().strip()
    if not prompt:
        ap.error("no prompt given (pass as arguments or on stdin)")

    bootstrap()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import transformers
    from transformers import AutoProcessor, TextStreamer

    if not torch.cuda.is_available():
        print("no GPU visible — this script needs one", file=sys.stderr)
        return 2

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    print(f"[gemma] {props.name}: {vram_gb:.0f} GB, sm{props.major}{props.minor}",
          file=sys.stderr)
    if vram_gb < MIN_VRAM_GB:
        print(f"[gemma] need >= {MIN_VRAM_GB:.0f} GB after decompression; "
              f"use the GGUF path in the notebook instead", file=sys.stderr)
        return 2

    ModelCls = (getattr(transformers, "AutoModelForMultimodalLM", None)
                or getattr(transformers, "AutoModelForImageTextToText", None)
                or transformers.AutoModelForCausalLM)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = ModelCls.from_pretrained(args.model, dtype="auto", device_map="auto")
    model.eval()
    print(f"[gemma] loaded {type(model).__name__} in {time.time() - t0:.0f}s",
          file=sys.stderr)

    content = [{"type": "text", "text": prompt}]
    if args.image:
        key = "url" if "://" in args.image else "path"
        content.insert(0, {"type": "image", key: args.image})  # image parts go first

    messages = [{"role": "system", "content": args.system},
                {"role": "user", "content": content}]

    template_kw = {"enable_thinking": args.thinking}
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, **template_kw)
    except TypeError:  # template on this build doesn't accept enable_thinking
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True)
    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    gen_kw = {}
    if args.temperature is not None:
        gen_kw.update(do_sample=True, temperature=args.temperature, top_p=0.95)

    streamer = TextStreamer(processor.tokenizer, skip_prompt=True,
                            skip_special_tokens=True)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             streamer=streamer, **gen_kw)

    n_new = out.shape[-1] - input_len
    dt = time.time() - t0
    print(f"\n[gemma] {n_new} tokens in {dt:.1f}s ({n_new / dt:.1f} tok/s), "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
