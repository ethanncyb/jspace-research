# Jacobian lens — Walkthrough

This document expands the notebook `walkthrough.ipynb` into a readable, section-by-section explanation with key code excerpts, rationale, and practical tips for running the examples.

---

## 1. Load the model

Purpose: load a HuggingFace causal LM, pick an execution device and dtype that match hardware, and wrap the HF model in `jlens`'s `LensModel` interface.

Key code excerpt:

```python
import torch
import transformers

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
    torch_dtype = torch.float16
else:
    device = torch.device("cpu")
    torch_dtype = torch.float32

hf_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype).to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf_model, tokenizer)
```

Notes & tips:
- Choose `torch_dtype` to match accelerator: `bfloat16` for CUDA A100/RTX 6000/A40 where supported, `float16` for MPS/half setups, `float32` on CPU to avoid OOM and dtype mismatches.
- `jlens.from_hf` wraps the HF model so later APIs (`lens.apply`, `compute_slice`, `steer`) can use a consistent interface.
- If you get device/dtype mismatch errors, try loading with `torch_dtype=None` and then `.to(device)`.

---

## 2. Load a pre-fitted lens

Purpose: load a saved Jacobian lens (one [d_model, d_model] matrix for each model layer). Lenses are typically stored as `.pt` files in a HF Hub repo or locally.

Key code excerpt:

```python
lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
```

Notes & tips:
- `LENS_REPO` may be a Hub repo like `neuronpedia/jacobian-lens` and `filename` points to the asset in that repo.
- You can also pass a local path to `from_pretrained` to load a lens saved on disk.
- After loading, `lens` contains per-layer Jacobian transport matrices and convenience methods such as `apply`, `steer`, and `save`.

---

## 3. Apply: J-lens vs logit lens

Purpose: produce logits that show how residuals at an intermediate layer map into final vocabulary logits. Two modes:
- J-lens (`use_jacobian=True`): transports residuals with the fitted Jacobian `J_l` into the final-layer basis and decodes with the model's own unembedding.
- Logit-lens (`use_jacobian=False`): decodes the layer residual directly through the unembedding without transport.

Key code excerpt:

```python
prompt = "Fact: The currency used in the country shaped like a boot is"
layers = [model.n_layers // 4, model.n_layers // 2, model.n_layers // 4 * 3, model.n_layers - 2]

jlens_logits, model_logits, _ = lens.apply(model, prompt, layers=layers, positions=[-2])
logit_lens, _, _ = lens.apply(model, prompt, layers=layers, positions=[-2], use_jacobian=False)

def top5(logits):
    return [tokenizer.decode([t]) for t in logits.topk(5).indices]

for layer in layers:
    print(f"L{layer:>3} logit-lens: {top5(logit_lens[layer][0])}")
    print(f"L{layer:>3} J-lens:     {top5(jlens_logits[layer][0])}")
print(f"model:           {top5(model_logits[0])}")
```

Notes & tips:
- Compare the printed top tokens. The J-lens often surfaces coherent, interpretable tokens earlier than the logit lens.
- `positions=[-2]` decodes for the token at offset `-2` relative to the prompt end; adjust as needed.
- If `lens.apply` raises a shape error, verify the model's `d_model` matches the lens matrix dimensions.

---

## 4. Render a slice page (inline)

Purpose: compute a position×layer view of the lens's token ranks and embed an interactive HTML page in the notebook (`mode="embed"` produces a self-contained page).

Key code excerpt:

```python
import gzip, json
from jlens.examples import EXAMPLES, resolve_prompt
from jlens.vis import build_page, compute_slice, notebook_iframe

gloss = {int(k): v for k, v in json.load(gzip.open("assets/qwen_gloss.json.gz")).items()}
example = next(e for e in EXAMPLES if e.slug == "multihop")
prompt = resolve_prompt(example, tokenizer)

slice_data = compute_slice(model, lens, prompt, layer_stride=2, mask_display=True)
page, _, _ = build_page(slice_data, prompt, title=example.section, description=example.description, alt_token=gloss)
notebook_iframe(page)
```

Notes & tips:
- `compute_slice` returns the per-layer rank/top-K data needed to render the interactive view.
- `mask_display=True` filters noisy short tokens (e.g., punctuation) to focus the display on word-like tokens.
- Use `layer_stride` to reduce the number of layers visualized for very deep models.

---

## 5. Render a slice page (served)

Purpose: for long prompts or large slices, write rank files to disk and serve them via a local HTTP server. The HTML page will lazily fetch rank files (`mode="fetch"`) to stay small.

Key code excerpt:

```python
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial
import threading, os

slice_data = compute_slice(model, lens, prompt, mask_display=True)
out_dir = Path("slices") / example.slug
page, _, _ = build_page(slice_data, prompt, title=example.section, description=example.description, alt_token=gloss, mode="fetch", out_dir=out_dir)
(out_dir / "index.html").write_text(page)

_handler = partial(SimpleHTTPRequestHandler, directory=os.path.abspath("slices"))
_jlens_httpd = HTTPServer(("127.0.0.1", 0), _handler)
threading.Thread(target=_jlens_httpd.serve_forever, daemon=True).start()
print(f"-> http://localhost:{_jlens_httpd.server_address[1]}/{example.slug}/")
```

Notes & tips:
- `mode="fetch"` writes rank files into `out_dir` and references them from `index.html`.
- The server uses an ephemeral port (`0`); the printed URL shows the actual port.
- Ensure `slices/` is in `.gitignore` if you plan to generate many large slice files.

---

## 6. Fitting

Purpose: compute the Jacobians `J_l` using a set of prompts. The fit procedure does backward passes in chunks (`dim_batch`) to control memory use.

Key code excerpt:

```python
from jlens.examples import load_wikitext_prompts

prompts = load_wikitext_prompts(n_prompts=100)
lens = jlens.fit(model, prompts, dim_batch=32, max_seq_len=128, checkpoint_path="ckpt.pt")
lens.save("jacobian_lens.pt")
```

Notes & tips:
- `n_prompts=100` is a practical minimum for a usable lens; larger (e.g., 1000) yields more stable transports.
- `dim_batch` controls how many model-output dimensions you compute per backward pass; increasing `dim_batch` reduces backward-pass count but increases memory.
- `checkpoint_path` lets the job resume or save intermediate progress.

---

## 7. JSpace next-token steering

Purpose: perform a small intervention by writing a chosen vocabulary token into the residual stream using the normalized row of `W_U J_\ell`. Measure how the intervention changes the gold next-token's rank (controllability).

Key code excerpt:

```python
humaneval_case = jlens.load_humaneval_cases(tokenizer, n_examples=1)[0]
gsm8k_case = jlens.load_gsm8k_cases(tokenizer, n_examples=1)[0]

for case in (humaneval_case, gsm8k_case):
    ids = torch.tensor([case.input_ids], dtype=torch.long, device=model.input_device)
    result = lens.steer(model, ids, target_token_id=case.target_token_id, strength=0.1)
    comparison = jlens.compute_steering_comparison(model, lens, result, last_n_tokens=32, mask_display=True)
    page = jlens.build_steering_comparison_page(comparison, title=f"{case.dataset}: gold next-token steering", description="Teacher-forced reference prefix; this is controllability, not task accuracy.")
    display(notebook_iframe(page, height=900))
    print("output target rank:", result.clean_target_ranks.item(), "->", result.steered_target_ranks.item())
```

Notes & tips:
- Steering scales interventions by the local clean residual norm so magnitudes are comparable across positions.
- The benchmark measures movement of the gold next-token's rank (smaller is better); this is a controllability metric, not an evaluation of task accuracy.
- Strength sweeps and matched random-direction controls (the optional 32+32 benchmark) provide statistical context for results.

---

## Running hints and troubleshooting

- If you see CUDA OOM errors, reduce `max_seq_len`, `dim_batch`, or run on CPU for smaller experiments.
- If tokenization yields unexpected characters, inspect `tokenizer.decode` outputs and consider `skip_special_tokens` options when decoding.
- For reproducible fitting / benchmarks, set seeds where provided by the library or wrap data-loading with fixed RNG states.

## File references

- Notebook: `walkthrough.ipynb` (interactive step-through examples).
- This guide: `walkthrough.md`.

---

If you want, I can also:
- Add this guide into a short README section and link it from `README.md`.
- Extract runnable example scripts under `scripts/` that perform a single end-to-end run (load model, load lens, compute slice).
