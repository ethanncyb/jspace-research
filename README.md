# jlens — Jacobian lens

> **Reference implementation.** Not maintained and not accepting contributions.

Companion code for [**Verbalizable Representations Form a Global Workspace in
Language Models**](https://transformer-circuits.pub/2026/workspace/index.html).

The Jacobian lens reads out what an internal activation is disposed to make the
model say. It linearly transports a residual-stream vector at any layer and
position into the final-layer basis, then decodes it with the model's own
unembedding into a ranked list of vocabulary tokens.

The transport is the average input–output Jacobian over a text corpus:

```
lens_l(h) = unembed( J_l @ h ), J_l = E[∂h_final / ∂h_l]
```

The expectation is over prompts, source positions, and all current-and-future
target positions in a generic web-text corpus; the precise estimator
(cotangents summed over target positions, then averaged over source positions)
is documented in the [`jlens.fitting`](jlens/fitting.py) module docstring.

This repo fits the lens on open-weights decoder transformers, applies it, and
renders the interactive layer × position view shown below. Examples use Qwen;
other HuggingFace decoders adapt cleanly.

![Slice visualisation: ASCII-face example](assets/slice_vis.png)

*The ASCII-face example: selecting the `^` (nose) position shows the lens
reading out "nose" at mid layers, although the word never appears in the
prompt.*

## Install

```bash
pip install -e .
```

## Usage

### Apply

To apply a pre-fitted lens:

```python
import torch
import transformers
import jlens

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
    torch_dtype = torch.float16
else:
    device = torch.device("cpu")
    torch_dtype = torch.float32

hf = transformers.AutoModelForCausalLM.from_pretrained(
    "org/model",
    torch_dtype=torch_dtype,
).to(device)
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-2],
)
for layer, logits in sorted(lens_logits.items()):
    print(layer, [tok.decode([t]) for t in logits[0].topk(5).indices])
```

### Steer a next token

J-space steering adds the target token's unit J-lens vector at selected
residual-stream sites. The strength is multiplied by the clean activation norm
at each site. On Qwen3.5-4B the default band is layers 10–26 and the default
position is the final prompt token.

```python
target_id = tok(" Italy", add_special_tokens=False).input_ids[0]
result = lens.steer(
    model,
    "Fact: The currency used in the country shaped like a boot is",
    target_token_id=target_id,
    strength=0.1,
)
print("rank:", result.clean_target_ranks.item(), "->", result.steered_target_ranks.item())

comparison = jlens.compute_steering_comparison(
    model, lens, result, last_n_tokens=32
)
page = jlens.build_steering_comparison_page(comparison)
```

The comparison page shows clean and steered layer × position readouts followed
by a heatmap of the target token's rank improvement. This is a causal
intervention, not a guarantee that the model has reasoned correctly.

### HumanEval and GSM8K controllability benchmark

With the `dev` extra installed, the benchmark helpers select deterministic
gold continuation tokens from teacher-forced reference solutions:

```python
cases = (
    jlens.load_humaneval_cases(tok, n_examples=32)
    + jlens.load_gsm8k_cases(tok, n_examples=32)
)
observations = jlens.run_gold_next_token_benchmark(model, lens, cases)
summary = jlens.summarize_benchmark(observations)
```

The reported top-1 rate, target rank, logit lift, and KL divergence measure
next-token controllability. They are not HumanEval pass@1 or GSM8K accuracy.

### Fit

To fit a lens on your own model:

```python
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

The paper's lenses use 1000 sequences of 128 tokens from a pretraining-like
corpus. Quality saturates quickly (§9.3); ~100 prompts is usable. This is a
reference implementation and is not optimized; fitting time is dominated by
the model's own backward pass. Parallelize by running `fit()` on disjoint
slices and combining with `JacobianLens.merge()`.

## Walkthrough

[`walkthrough.ipynb`](walkthrough.ipynb) is the end-to-end notebook: load a
model and lens, inspect layer × position slices, steer a gold next token,
compare clean and intervened J-space activity, and optionally run the
HumanEval/GSM8K benchmark.

Reading a slice page:

- Each cell shows the lens top-1 word at that (position, layer); the
  superscript is its rank over the full vocabulary.
- Click a cell to select a (position, layer) and pin its top-1 token; pinned
  tokens get rank-tracking charts and a rank heatmap.
- The bottom row (`L = n_layers − 1`) is the model's actual output.

## Security probe

A small inference-time guard inspired by DARWIN-Guard: instead of updating
weights, the probe learns by appending jailbreak attempts to a persistent
experience memory and scoring new prompts against it. Three modules:

- [`jlens/guard_embed.py`](jlens/guard_embed.py) — deterministic hashed
  character n-gram embeddings, L2-normalized.
- [`jlens/guard_memory.py`](jlens/guard_memory.py) — a JSONL-backed
  `GuardMemory` of `Incident` records with filtered similarity search.
- [`jlens/guard_probe.py`](jlens/guard_probe.py) — `SecurityProbe`: a prompt
  whose nearest memory match clears the block threshold is blocked (a lower
  repeat-offender threshold applies when the match is a past success by the
  same user), close matches are flagged, and the rest are allowed.

The mock benchmark in
[`data/experiments/security-probe.json`](data/experiments/security-probe.json)
pairs non-actionable harmful-intent stubs with a strategy vulnerability map
and benign lookalikes. Run the demo with:

```bash
python scripts/security_probe_demo.py
```

## License and data

Code is released under the Apache License 2.0 — see [LICENSE](LICENSE).

The replication and lens-eval prompt sets in [`data/`](data/) are synthetic,
authored by Anthropic, and released under the same Apache License 2.0 as the
code. See the READMEs in [`data/experiments/`](data/experiments/) and
[`data/evaluations/`](data/evaluations/) for what each set contains.

The slice-vis pages use [d3](https://github.com/d3/d3) (ISC license), loaded
from the jsDelivr CDN with subresource integrity or inlined into
self-contained pages.

No model weights or text corpora are bundled; models and datasets downloaded
at run time are subject to their own licenses.
