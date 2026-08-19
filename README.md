# jspace-research

Use WSL FOR THIS

Gemma 4 12B QAT inference on Colab GPUs, plus Jacobian-lens fitting
([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)).

## Quickstart — run inference and watch the lens

One command. It provisions an A100, uploads the lens, loads the model, and serves the
panel locally:

```bash
python3 scripts/lens_panel.py --session lens
```

Then open **http://127.0.0.1:8765**.

In the panel you can set a **system prompt**, edit the **conversation history** (add and
remove user/assistant turns), give the assistant a **prefill** to read out mid-answer,
attach an **image**, and switch between a prompt slice and stepping through generation.
Hit **Run lens** and the layer x position grid appears. Press **Stop Colab session** when
you're done — idle A100s burn compute units and nothing else reclaims them.

First run takes a couple of minutes (VM provisioning + model load). After that a text
slice is ~1 s and an image slice ~2 s, because the model stays resident in the kernel.

The only prerequisite is an authenticated `colab` CLI (see below). The lens itself is
pulled onto the VM from
[PxlNexus/gemma4-JLens](https://huggingface.co/PxlNexus/gemma4-JLens),
so a fresh clone needs no local 649 MB binary.

> **Use the chat template.** This checkpoint is channel/thinking-tuned and collapses on
> raw continuation: `"The capital of France is"` greedily continues to `"111"`, and the
> final lens row (L47, J = identity, i.e. the model's own output) reads out `' shaped'`
> for a raw fact prompt. Templated, the same prompt gives `L42 'euro'` → `L46 'Euro'` →
> `L47 'The'/'Italy'/'Euro'`. The panel defaults to the chat template for this reason;
> "raw text" is there for base-model-style analysis.

## Layout

| Path | What |
|---|---|
| `notebooks/gemma4_12b_qat_inference.ipynb` | Download + inference, with a GGUF fallback for small GPUs |
| `scripts/gemma4_infer.py` | One-shot inference; self-cleaning via `colab run` |
| `scripts/fit_gemma4_lens.py` | Fit a Jacobian lens on the QAT model |
| `scripts/apply_gemma4_lens.py` | Read out what each layer is "disposed to say" |
| `scripts/lens_slice_service.py` | VM-side: keeps model + lens warm, returns slice payloads |
| `scripts/lens_panel.py` | Local interactive panel (runs on your machine) |

## Colab CLI

Auth is ADC and the flag goes **before** the subcommand:

```bash
gcloud auth application-default login --no-launch-browser \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
colab --auth=adc sessions
```

The `colaboratory` scope is what stops the keep-alive service 403ing.

```bash
colab --auth=adc new  -s gemma --gpu A100
colab --auth=adc run  --gpu A100 --timeout 2400 scripts/gemma4_infer.py "your prompt"
colab --auth=adc stop -s gemma       # nothing else reclaims the VM
```

**Always pass `--timeout`.** Every exec-ish command defaults to **30 seconds**, which
is shorter than the weight download.

## Three things that will bite you

**1. `w4a16-ct` is 4-bit on disk and BF16 in VRAM.** `compressed-tensors` 0.18.0 does
not run `pack-quantized` weights packed — a hook decompresses every `Linear` to BF16 on
the first forward. The checkpoint loads at 8.3 GB and then runs at ~25 GB peak.
`run_compressed=True` does not change this. So a 24 GB L4 loads it fine and dies on the
first token; you need ~30 GB, i.e. an A100. Below that, use the notebook's GGUF path,
which really does stay ~7 GB.

**2. Never trigger that decompression under `torch.inference_mode()`.** The
materialised weights become *inference tensors*, and every later autograd call fails
with `Inference tensors cannot be saved for backward` — which kills lens fitting
outright. Warm the model up with a `torch.no_grad()` forward instead. Plain inference
is unaffected; this only matters if you later need gradients.

**3. The kernel persists between `colab exec` calls.** Loading a model twice in one
session OOMs. `colab restart-kernel -s <name>` before re-running a script that loads
weights.

## Jacobian lens

d_model 3840, 48 layers, and jlens auto-detects the text decoder at
`model.language_model`. Fitting *all* layers costs the same GPU time as fitting one —
every layer's gradient comes out of the same backward pass — so only host RAM and
checkpoint size grow with layer count.

Cost on an A100-40GB: one forward + `ceil(3840 / dim_batch)` backward passes per prompt.
Measured 73 s/prompt at `dim_batch=16` (peak 40.2 GB, too close to the edge for an
all-layer fit) — hence the `dim_batch=8` default.

```bash
colab --auth=adc new -s lens --gpu A100
LENS_ARGS="--n-prompts 100" \
  colab --auth=adc exec -s lens --timeout 36000 -f scripts/fit_gemma4_lens.py
colab --auth=adc download -s lens /content/gemma4_12b_qat_lens.pt
colab --auth=adc stop -s lens
```

`colab exec -f` runs the file *inside the IPython kernel*, so `sys.argv` is the
kernel's, not yours — pass options via `LENS_ARGS`. Under `colab run` they are ordinary
arguments. The fit checkpoints and resumes, so a dropped session is not lost work.

For anything long, launch it **detached on the VM** rather than holding an exec open —
`exec --timeout` bounds only the *client* wait, and a dropped websocket kills your view
of the job but not the job:

```python
subprocess.Popen(f"nohup {sys.executable} /content/fit.py ... > /content/fit.log 2>&1 &",
                 shell=True, start_new_session=True)
```

Measured: **95 s/prompt** for 22 source layers at `dim_batch=8` (peak 34.6 GB), so
100 prompts is roughly 2.6 h.

## Interactive panel

Type a prompt, optionally attach an image, watch the layer x position readout. The GPU
work runs on Colab; the view renders in your local browser. No tunnel and no public URL
— the transport is `colab exec` against a warm kernel.

```bash
colab --auth=adc new -s lens --gpu A100
python3 scripts/lens_panel.py --session lens     # then open http://127.0.0.1:8765
```

The first run uploads the lens and loads the model into the kernel (a couple of
minutes); after that a text slice takes ~1 s and an image slice ~2 s, because the model
stays resident between calls. The panel has a **Stop Colab session** button — idle A100s
burn compute units, and nothing else reclaims them.

The visualisation is jlens's own page in `mode="fetch"`, which reads `meta.json` /
`slice.bin` / `ranks/*.bin` from `?datapath=`. That page is slice-independent, so it is
cached once and repointed per run; the panel serves the sidecar files from
`panel/runs/<id>/` on the same origin (the page rejects cross-origin datapaths).

**Images.** jlens's stock `HFLensModel.forward` calls the bare text decoder, which does
no image merging, so images would be invisible to it. `Gemma4LensModel` instead encodes
through the processor's chat template and forwards through `model.model`, the unified
module that merges vision embeddings before running the same decoder blocks the hooks
sit on. An attached photo shows up as ~260 `<|image|>` positions on the slice axis — use
"last N tokens" to focus the grid on the text.

**Bound `max_tracked` in generation mode.** `compute_slice` defaults it to `None`, which
keeps a full rank tensor for *every* token appearing in any top-K cell — ~3000 for a short
prompt, each written as its own `ranks/{tid}.bin`. Multiplied by the step count that
produced a **67 MB payload of ~24k files** which crawls back through the exec stdout
bridge, even though the GPU work took 20 s. Capping at 40 gives **8.1 s / 1.4 MB /
337 files** for the same 8 steps. `slice_payload` caps at 60; `generation_payload` now
caps at 40.

**Generation generates once, then slices prefixes.** Decoding one token at a time by
re-running `generate()` over the whole sequence recomputes attention from scratch rather
than using the KV cache, and in BF16 that shifts logits enough to flip a near-tied
argmax: the panel showed *"...country shaped**1** like a boot..."* where the model
actually says *"...country shaped like a boot..."*. One `generate()` call is both
faithful and cheaper. Slices are then written for prefixes of the real output, every
`slice_every` steps (derived from `max_slices=16`), so 96 tokens costs about the same as
16 — measured **6.9 s / 1.0 MB / 211 files**.

Note the default `generation_config` has `do_sample=True`; pass `do_sample=False`
explicitly for reproducible greedy decoding.

**Generation stepping** runs through the chat template. Raw greedy continuation collapses
on this checkpoint ("The capital of France is" → "111") because it is channel/thinking-
tuned; templated, it answers "…is **Paris**." Slices are computed on exact token ids
rather than re-tokenized text, so control and image tokens round-trip faithfully.

**The lens comes from the Hub, not from you.** Colab reclaims idle VMs — a 404/401 on
exec means the backend pruned yours, and `/content` went with it — so the lens has to
land again on every boot. Pushing it costs your uplink; pulling it costs Colab's:

| Direction | Measured |
|---|---|
| push, `upload_large` from WSL | 649 MB in **~170 s** (~4.7 MB/s uplink, +33% base64) |
| pull, `hf_hub_download` on the VM | 649 MB in **4 s** (**181 MB/s**) |

That is a **42x** difference, and it is not a fluke: the push is bounded by a home
uplink, the pull by a datacentre one. The panel downloads from
[PxlNexus/gemma4-JLens](https://huggingface.co/PxlNexus/gemma4-JLens)
and verifies sha256 against a pinned digest before installing it, so a truncated
transfer or a changed repo fails loudly instead of loading a corrupt lens.
`upload_large()` remains as an automatic fallback when the fetch fails.

```bash
python3 scripts/lens_panel.py --session lens                  # pull from the Hub
python3 scripts/lens_panel.py --hf-repo you/your-own-lens     # a different lens
python3 scripts/lens_panel.py --no-hf                         # push panel/*.pt instead
```

`--hf-repo` clears the pinned hash, since another repo will not match it. To publish a
freshly fitted lens, `hf upload <repo> gemma4_12b_qat_lens.pt` and update
`Config.hf_sha256`.

### The exec socket stalls sometimes

`colab exec --timeout` is a **client-side** wait only — the kernel keeps running
regardless. When the websocket stalls (this project has hit
`WebSocketConnectionClosedException` mid-call), the CLI blocks for the whole budget and
then reports only:

```
Command '['colab', '--auth=adc', 'exec', '-s', 'lens', '--timeout', '900']'
timed out after 1020 seconds
```

Two things the panel now does about it:

- **Budgets sized to the work**, so a stall surfaces in minutes rather than 17 of them:
  240 s for a slice (measured ~1 s text / ~2 s image), `120 + 30/step` for generation
  (8 steps measured at 8.1 s), 1800 s only for a cold boot.
- **`exec_code(..., retries=1)`** retries a stalled call, with an error message that says
  what to do instead of dumping the raw subprocess exception.

Retries are opt-in per call site because they are not universally safe. The chunk **join
is idempotent** for exactly this reason: a stalled attempt that already finished on the
VM will have deleted the `.part` files, so a naive re-run would find no parts and
truncate the target to zero bytes. The joiner detects "no parts but the file exists" and
re-hashes what is there instead.

VM disk is not the bottleneck — measured **2127 MB/s write, 337 MB/s read+sha256**, so a
20-part join is ~2 s of I/O.

### `colab upload` cannot handle large files

Only the fallback path hits this now — but it is still the thing to know if you ever
push a big file to a VM.

`colab_cli/contents.py` reads the whole file, base64-encodes it, and PUTs it as a single
JSON body with a hardcoded `"chunk": 1`:

```python
content = f.read()
b64_content = base64.b64encode(content).decode("ascii")
payload = {..., "content": b64_content, "chunk": 1}
```

Measured ceiling on a CPU runtime: **64 MB uploads fine (19 s), 128 MB returns 500,
256 MB returns 400.** A 649 MB lens therefore fails with

```
Upload failed: 500 Server Error: Internal Server Error for url: .../api/contents/content/...
```

`lens_panel.upload_large()` works around it by splitting into 32 MB parts (~6 s each,
one on disk at a time), uploading them as `<remote>.partNNNN`, concatenating on the VM,
and verifying size + sha256 before deleting the parts. The 649 MB lens takes ~170 s in
20 chunks. The Jupyter contents API does support real chunked uploads (`chunk: 1` writes,
`2..n` append, `-1` finalizes) — the CLI just never uses them.

Fitting fewer source layers is the way to shrink this: the lens is
`len(source_layers) x 3840^2` fp16, so 22 layers = 649 MB and halving the layers halves
the upload.
