#!/usr/bin/env python3
"""VM-side slice service for the local lens panel.

Loaded ONCE into a persistent Colab kernel:

    colab --auth=adc exec -s lens --timeout 1800 -f scripts/lens_slice_service.py

That boots the model + lens and leaves `slice_payload` / `generation_payload` defined
in the kernel. Kernel state persists across `colab exec` calls, so the local panel
(scripts/lens_panel.py) then issues cheap one-liners against the warm model:

    print(slice_payload("Fact: the capital of France is"))

Each payload is a base64 tar.gz of the sidecar files a jlens `mode="fetch"` page
reads (meta.json, slice.bin, ranks/*.bin), fenced by sentinels so the caller can
pick it out of Colab's noisy stdout.

Gemma 4 vision
--------------
jlens's stock HFLensModel.forward calls `model.language_model`, the bare text
decoder, which performs no image merging -- so images are invisible to it. The
subclass below instead runs the processor's chat template (which expands an image
into real image tokens) and calls `model.model`, the unified module that merges
vision embeddings and then runs the same decoder blocks. The ActivationRecorder
hooks sit on those block objects, so they fire either way and image positions show
up on the slice's position axis like any other token.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

BEGIN = "<<<JLENS_PAYLOAD_BEGIN>>>"
END = "<<<JLENS_PAYLOAD_END>>>"

MODEL_ID = os.environ.get("LENS_MODEL", "google/gemma-4-12B-it-qat-w4a16-ct")
LENS_PATH = os.environ.get("LENS_PATH", "/content/gemma4_12b_qat_lens.pt")

BOOTSTRAP = {
    "compressed_tensors": "compressed-tensors",
    "hf_transfer": "hf_transfer",
    "jlens": "git+https://github.com/anthropics/jacobian-lens.git",
}


def _bootstrap() -> None:
    import importlib.util

    missing = [pkg for mod, pkg in BOOTSTRAP.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"[svc] installing {' '.join(missing)}", file=sys.stderr, flush=True)
    for installer in (["uv", "pip", "install", "--system", "-q"],
                      [sys.executable, "-m", "pip", "install", "-q"]):
        if subprocess.run(installer + missing).returncode == 0:
            return
    raise SystemExit("[svc] dependency install failed")


_bootstrap()

import torch  # noqa: E402
import transformers  # noqa: E402
import jlens  # noqa: E402
from jlens.hf import HFLensModel  # noqa: E402
from jlens.vis import compute_slice, write_slice_files  # noqa: E402


class Gemma4LensModel(HFLensModel):
    """LensModel that can carry an image alongside the text prompt.

    Text-only prompts take the stock path (raw tokenizer + text decoder) so the
    readout matches how the lens was fitted. When an image is attached, encoding
    switches to the processor's chat template and the forward goes through the
    unified module so vision tokens are merged.
    """

    def __init__(self, hf_model, processor, **kw) -> None:
        super().__init__(hf_model, processor.tokenizer, **kw)
        self.processor = processor
        self._image = None
        self._extra: dict[str, torch.Tensor] = {}
        self._forced: tuple[torch.Tensor, dict] | None = None

    def set_image(self, image) -> None:
        """Attach a PIL image (or None) for subsequent encode/forward calls."""
        self._image = image
        self._extra = {}

    def set_forced(self, input_ids: torch.Tensor | None,
                   extra: dict | None = None) -> None:
        """Pin the exact token ids `encode` should return.

        Generation stepping needs slices over a sequence that already contains
        chat-template control tokens and (for images) expanded image tokens.
        Re-tokenizing a decoded string would not round-trip those faithfully, so
        the caller hands us the ids directly.
        """
        self._forced = None if input_ids is None else (input_ids, dict(extra or {}))

    @property
    def has_image(self) -> bool:
        return self._image is not None

    def chat_encode(self, text: str, image=None) -> tuple[torch.Tensor, dict]:
        """Single-user-turn convenience wrapper over :meth:`build_inputs`."""
        ids, extra, _ = self.build_inputs([{"role": "user", "content": text}],
                                          image=image)
        return ids, extra

    def build_inputs(
        self,
        messages: list[dict],
        *,
        image=None,
        chat: bool = True,
        assistant_prefill: str = "",
        add_generation_prompt: bool = True,
        max_seq_len: int = 512,
    ) -> tuple[torch.Tensor, dict, str]:
        """Turn a message list into (input_ids, extra, display_text).

        `chat=True` runs the processor's chat template, which is what this
        checkpoint actually expects -- raw continuation of a bare fragment is
        out-of-distribution and collapses (see module docstring of
        fit_gemma4_lens.py and the README). `chat=False` concatenates the message
        contents and tokenizes them raw, for base-model-style analysis.

        An image, if given, is attached to the last user message. An
        `assistant_prefill` is appended after the generation prompt so the readout
        can be taken mid-assistant-turn.
        """
        device = self.input_device

        if not chat:
            text = "\n".join(m.get("content", "") for m in messages)
            return super().encode(text, max_length=max_seq_len), {}, text

        last_user = max((i for i, m in enumerate(messages)
                         if m.get("role") == "user"), default=None)
        conv: list[dict] = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if image is not None and i == last_user:
                conv.append({"role": msg["role"], "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": content}]})
            else:
                conv.append({"role": msg["role"], "content": content})

        def render(conversation: list[dict]):
            return self.processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=add_generation_prompt)

        try:
            enc = render(conv)
        except Exception:
            # Some Gemma chat templates reject a standalone system turn; fold it
            # into the first user message instead of failing.
            system = " ".join(m.get("content", "") for m in conv
                              if m.get("role") == "system").strip()
            rest = [m for m in conv if m.get("role") != "system"]
            if not system or not rest:
                raise
            head = rest[0]
            if isinstance(head["content"], list):
                merged = list(head["content"])
                for part in merged:
                    if part.get("type") == "text":
                        part["text"] = f"{system}\n\n{part['text']}"
                        break
            else:
                merged = f"{system}\n\n{head['content']}"
            rest[0] = {"role": head["role"], "content": merged}
            enc = render(rest)

        input_ids = enc["input_ids"].to(device)
        extra = {k: v.to(device) for k, v in enc.items()
                 if k != "input_ids" and torch.is_tensor(v)}
        extra.pop("attention_mask", None)   # regenerated as the sequence grows

        if assistant_prefill:
            pre = self.tokenizer(assistant_prefill, add_special_tokens=False,
                                 return_tensors="pt").input_ids.to(device)
            input_ids = torch.cat([input_ids, pre], dim=-1)

        return input_ids, extra, self.tokenizer.decode(input_ids[0],
                                                       skip_special_tokens=False)

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        if self._forced is not None:
            input_ids, self._extra = self._forced
            return input_ids
        if self._image is None:
            self._extra = {}
            return super().encode(text, max_length=max_length)
        input_ids, self._extra = self.chat_encode(text, self._image)
        return input_ids

    def forward(self, input_ids: torch.Tensor):
        if not self._extra:
            return super().forward(input_ids)
        # `model.model` merges vision embeddings, then runs the same decoder
        # blocks the recorder is hooked onto.
        return self._hf_model.model(
            input_ids=input_ids, use_cache=False, **self._extra)


def boot(model_id: str = MODEL_ID, lens_path: str = LENS_PATH):
    """Load model + lens into the kernel. Idempotent.

    Re-running this file (to pick up edits) always rebuilds the `lens_model`
    wrapper: redefining the class leaves any existing instance bound to the OLD
    class object, so it would silently lack newly added methods. The expensive
    parts -- weights and lens -- are reused.
    """
    global model, processor, lens, lens_model

    if globals().get("_booted_with") == (model_id, lens_path) and "model" in globals():
        lens_model = Gemma4LensModel(model, processor)
        print("[svc] weights already resident; lens_model wrapper rebuilt",
              file=sys.stderr)
        return lens_model

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    processor = transformers.AutoProcessor.from_pretrained(model_id)
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype="auto", device_map="auto")
    model.eval()
    # Materialise decompressed BF16 weights OUTSIDE inference_mode; see README.
    with torch.no_grad():
        model(input_ids=torch.tensor([[2, 108, 109]], device=model.device),
              use_cache=False)

    lens = jlens.JacobianLens.load(lens_path)
    lens_model = Gemma4LensModel(model, processor)
    globals()["_booted_with"] = (model_id, lens_path)

    print(f"[svc] {lens_model}", file=sys.stderr)
    print(f"[svc] lens: {len(lens.source_layers)} layers, n_prompts={lens.n_prompts}",
          file=sys.stderr)
    print(f"[svc] VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB", file=sys.stderr)
    return lens_model


def _image_from_b64(image_b64: str | None):
    if not image_b64:
        return None
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _pack(out_dir: Path, extra_meta: dict) -> str:
    """tar.gz the sidecar dir and return base64, fenced by sentinels."""
    (out_dir / "panel.json").write_text(json.dumps(extra_meta), encoding="utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out_dir, arcname=".")
    return BEGIN + base64.b64encode(buf.getvalue()).decode() + END


def _as_messages(prompt: str | None, messages: list[dict] | None,
                 system: str | None) -> list[dict]:
    """Normalise the panel's fields into a chat message list."""
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    if messages:
        out.extend({"role": m.get("role", "user"), "content": m.get("content", "")}
                   for m in messages if m.get("content", "").strip()
                   or m.get("role") == "assistant")
    elif prompt is not None:
        out.append({"role": "user", "content": prompt})
    if not any(m["role"] != "system" for m in out):
        out.append({"role": "user", "content": prompt or ""})
    return out


def slice_payload(
    prompt: str | None = None,
    image_b64: str | None = None,
    *,
    messages: list[dict] | None = None,
    system: str | None = None,
    assistant_prefill: str = "",
    chat: bool = True,
    top_n: int = 10,
    max_tracked: int = 60,
    layer_stride: int = 1,
    last_n_tokens: int | None = None,
    max_seq_len: int = 512,
    mask_display: bool = False,
    title: str = "Gemma 4 12B QAT - Jacobian lens",
) -> str:
    """Compute one layer x position slice and return it as a fenced base64 tarball.

    `max_tracked` is capped by default because the rank sidecars dominate payload
    size (roughly len(tracked) * seq_len * n_layers * 4 bytes before compression)
    and every byte here crosses the `colab exec` stdout bridge.
    """
    image = _image_from_b64(image_b64)
    convo = _as_messages(prompt, messages, system)
    lens_model.set_forced(None)          # drop any state left by a generation run
    lens_model.set_image(None)
    try:
        input_ids, extra, display = lens_model.build_inputs(
            convo, image=image, chat=chat, assistant_prefill=assistant_prefill,
            max_seq_len=max_seq_len)
        lens_model.set_forced(input_ids, extra)
        slice_data = compute_slice(
            lens_model, lens, display,
            top_n=top_n, max_tracked=max_tracked, layer_stride=layer_stride,
            last_n_tokens=last_n_tokens, max_seq_len=max_seq_len,
            mask_display=mask_display)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "slice"
            write_slice_files(
                slice_data, out, prompt=display, title=title,
                description=("chat template" if chat else "raw text")
                + (" + image" if image is not None else ""))
            # Top tokens at the last position of the FINAL layer. That row is
            # J = identity, i.e. the model's own next-token distribution, so it is
            # the honest "did inference actually run" signal for the panel.
            final_top = [slice_data.vocab_fragment.get(int(t), "?")
                         for t in slice_data.top_ids[-1, -1, :5]]
            return _pack(out, {
                "kind": "slice",
                "seq_len": slice_data.seq_len,
                "layers": slice_data.layers,
                "has_image": image is not None,
                "chat": chat,
                "n_messages": len(convo),
                "n_tracked": len(slice_data.tracked_token_ids),
                "final_top": final_top,
            })
    finally:
        lens_model.set_forced(None)
        lens_model.set_image(None)


def generation_payload(
    prompt: str | None = None,
    image_b64: str | None = None,
    *,
    messages: list[dict] | None = None,
    system: str | None = None,
    assistant_prefill: str = "",
    chat: bool = True,
    max_new_tokens: int = 64,
    slice_every: int | None = None,
    max_slices: int = 16,
    max_tracked: int = 40,
    **slice_kw,
) -> str:
    """Generate token by token, computing a slice after each one.

    Runs through the chat template, not raw continuation: this checkpoint is
    channel/thinking-tuned and greedy-continuing a bare fragment collapses into
    degenerate output ("The capital of France is" -> "111"), while the templated
    form answers correctly. The slice therefore covers the full templated
    sequence, control tokens included.

    Generation is cheap (~19 tokens in 2.2 s); the slices are what cost. So every
    token is generated, but a slice is only written every `slice_every` steps --
    derived from `max_slices` when not given, so asking for 200 tokens costs about
    the same as asking for 16. Without this, "generate 80 tokens" meant 80 full
    readouts and a payload to match.

    Returns a tarball whose panel.json carries every generated token in `steps`,
    the step indices that actually have a slice directory in `slices`, and the
    finished answer in `final_text`.

    `max_tracked` must be bounded here. `compute_slice` defaults it to None, which
    keeps a full rank tensor for *every* token appearing in any top-K cell -- around
    3000 of them for a short prompt, each written as its own ranks/{tid}.bin. Times
    the step count that produced a 67 MB payload of ~24k files, which crawls through
    the exec stdout bridge even though the GPU work took 20 s.
    """
    image = _image_from_b64(image_b64)
    tokenizer = processor.tokenizer
    input_ids, extra, _ = lens_model.build_inputs(
        _as_messages(prompt, messages, system), image=image, chat=chat,
        assistant_prefill=assistant_prefill,
        max_seq_len=slice_kw.get("max_seq_len", 512))
    prompt_len = input_ids.shape[-1]
    steps: list[dict] = []

    if slice_every is None:
        slice_every = max(1, -(-max_new_tokens // max(1, max_slices)))
    sliced: list[int] = []

    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gen"
            root.mkdir(parents=True)

            def write_step(step: int, ids: torch.Tensor, extra: dict) -> None:
                lens_model.set_forced(ids, extra)
                text = tokenizer.decode(ids[0], skip_special_tokens=False)
                data = compute_slice(lens_model, lens, text,
                                     max_tracked=max_tracked, **slice_kw)
                write_slice_files(data, root / f"step{step:03d}", prompt=text,
                                  title=f"step {step}",
                                  description=f"after {step} generated token(s)")
                sliced.append(step)

            def grow(base: dict, length: int) -> dict:
                """Extend per-position extras (mm_token_type_ids) to `length`.

                Generated positions are text, so they take type 0."""
                out = {}
                for key, value in base.items():
                    if torch.is_tensor(value) and value.dim() >= 2 \
                            and value.shape[1] < length:
                        pad = torch.zeros(
                            (value.shape[0], length - value.shape[1], *value.shape[2:]),
                            dtype=value.dtype, device=value.device)
                        out[key] = torch.cat([value, pad], dim=1)
                    else:
                        out[key] = value
                return out

            # Generate ONCE, then slice prefixes of the real output. Decoding one
            # token at a time by re-running generate() over the whole sequence
            # recomputes attention from scratch instead of using the KV cache; in
            # BF16 that shifts logits just enough to flip a near-tied argmax, and
            # the panel showed "...country shaped1 like a boot..." where the model
            # actually says "...country shaped like a boot...". One pass is both
            # faithful and far cheaper.
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    **extra, max_new_tokens=max_new_tokens, do_sample=False)

            new_ids = out[0, prompt_len:].tolist()
            for i, token_id in enumerate(new_ids):
                steps.append({
                    "step": i,
                    "token": tokenizer.decode([token_id], skip_special_tokens=False),
                    "seq_len": prompt_len + i,
                })

            # step k == the state after k generated tokens; always include the end.
            wanted = sorted(set(range(0, len(new_ids), slice_every)) | {len(new_ids)})
            for k in wanted:
                ids_k = out[:, :prompt_len + k]
                write_step(k, ids_k, grow(extra, ids_k.shape[1]))

            input_ids = out
            generated = tokenizer.decode(input_ids[0, prompt_len:],
                                         skip_special_tokens=True)
            return _pack(root, {"kind": "generation", "steps": steps,
                                "slices": sorted(sliced),
                                "slice_every": slice_every,
                                "final_text": generated,
                                "has_image": image is not None})
    finally:
        lens_model.set_forced(None)
        lens_model.set_image(None)


if __name__ == "__main__":
    boot()
    print("[svc] ready: slice_payload(prompt, image_b64=None) / generation_payload(...)",
          file=sys.stderr)
