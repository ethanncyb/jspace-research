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

## Qwen3 model-size study (Colab)

[`qwen_size_study.ipynb`](qwen_size_study.ipynb) is a resumable Colab
experiment comparing the dense Qwen3 4B, 8B, 14B, and 32B checkpoints. The
4B/8B/14B runs target an L4; the 32B run requires an A100 with at least 38 GB
usable VRAM. Every model uses frozen NF4 weights with BF16 compute so precision
does not change with model size.

When the notebook is opened in Cursor with a remote Colab kernel, that kernel
cannot see the local Cursor checkout. Commit and push
`codex/qwen-model-size-test` first. The notebook's initial bootstrap cell clones
that branch into `/content/jspace-research`, installs it editable, and verifies
`import jlens` before mounting Drive or loading a model. If you intentionally
use another remote or branch, change `REPO_URL` and `REPO_BRANCH` in that first
cell.

Install the notebook dependencies with:

```bash
pip install -e '.[study]'
```

The notebook has a single `ACTIVE_MODEL` toggle and separate `smoke` and
`full` profiles. Full behavioral evaluation uses all 164 HumanEval and 1,319
GSM8K test cases. The more expensive oracle controllability track uses a fixed
64-case subset from each dataset and the normalized strength sweep
`0, 0.025, 0.05, 0.1, 0.2, 0.4` at 25%, 50%, and 75% model depth.

The scalable local quantity is exact for one prompt and target token:

```text
g[layer, position] = d target_logit / d residual[layer, position]
```

`jlens.compute_local_jacobian` captures that sensitivity and
`jlens.steer_local` applies a clean-residual-norm-scaled intervention. Matched
random directions provide the causal control. The generated HTML pages align a
token-by-layer sensitivity heatmap with a layer-by-strength random-adjusted
target-rank heatmap. These views measure sensitivity and controllability; they
are not a decoder of private thoughts.

Security evaluation uses the distributable email, table, and code tasks from
[Microsoft BIPIA](https://github.com/microsoft/BIPIA). It reports explicit
`BENIGN`/`INJECTION` self-report, fixed-capacity PCA-128 residual probes at
eight relative-depth checkpoints, harmless-canary attack success, and clean
task utility as separate outcomes. It does not activate the circuit breaker or
continual-learning prototype below.

Each model writes a manifest and independent `benchmarks/`, `steering/`,
`jspace/`, and `security/` artifacts beneath a shared experiment ID on Google
Drive. JSONL is the append-only resume source; Parquet/CSV files are analysis
exports. Model and dataset revisions, seeds, generation settings, package
versions, GPU/VRAM, quantization, truncation, and git commit are recorded before
results are compared. Four size points support an exploratory trend report, not
a scaling-law claim.

## Qwen 3.5 residual prompt-injection prototype

The repository also contains an experimental, prefill-only prompt-injection
detector. It compares the last-token residual from a trusted clean prefix with
the last-token residual from the exact same prefix tokens plus a separately
encoded, untrusted segment. One logistic probe is trained at each configured
full-attention checkpoint. Qwen's GDN layers may be observed, but hooks never
replace their outputs.

The default checkpoints are zero-based layers `3, 7, 11, 15, 19, 23, 27, 31`.
They are checked against `text_config.layer_types` when that field is present;
the hidden size comes from the model configuration, so the same modules work
with Qwen 3.5 4B (2560) and 9B (4096). Generated tokens are not scored or
modified in v1.

### Dataset contract

Input may be a JSON array, JSONL, or CSV. Every row has:

```json
{
  "id": "pair-001-injected",
  "clean_prompt": "Trusted system and user prefix",
  "appended_text": " untrusted suffix",
  "label": "injected",
  "pair_id": "pair-001"
}
```

Labels may be `0`/`1` or `benign`/`injected`. Column names can be remapped with
`field_map` in `config.yaml`. A row may provide `candidate_prompt` instead of
`appended_text`; this requires a tokenizer, and is accepted only if the clean
token IDs are an exact prefix of the candidate token IDs. `pair_id` is optional
metadata, but using it prevents related examples from leaking across the
grouped train/validation split.

### Workflow

Install the package, including the explicit YAML dependency, and collect
features:

```bash
pip install -e .
python eval_harness.py --config config.yaml collect data.jsonl \
  --output outputs/features.pt
```

Train, evaluate, and perform guarded deterministic generation:

```bash
python eval_harness.py --config config.yaml train outputs/features.pt \
  --output outputs/probe.pt
python eval_harness.py --config config.yaml evaluate \
  outputs/features.pt outputs/probe.pt --output-dir outputs/evaluation
python eval_harness.py --config config.yaml generate outputs/probe.pt \
  "trusted prefix" " appended segment"
```

Add `--exercise-intervention` to `evaluate` to rerun each pair through the
configured circuit-breaker or hard-stop path and populate guarded outcomes.

`run` performs collection, training, and evaluation end to end. Evaluation
writes `per_layer.csv`, `per_example.csv`, and `summary.csv`, plus the
continual-learning auto-positive JSONL buffer and manual-review CSV. Probe
checkpoints include normalization statistics, benign reference vectors,
raw-space attack directions, per-layer thresholds, layer indices, hidden size,
aggregation, and model ID. Incompatible checkpoints and malformed prompt pairs
fail before evaluation.

Circuit-breaker mode projects the learned raw probe direction out of every
appended-token residual at a triggering checkpoint. Later checkpoints therefore
observe the already-corrected stream. Hard-stop mode interrupts prefill,
discards partial model output/cache, and returns the configured refusal.
Set `intervention.mode` to `disabled`, `circuit_breaker`, or `hard_stop`.

To switch to 9B, change `model_id` in `config.yaml`; no hidden-size constant
needs editing. The 4B model is the more practical development default. CPU is
supported but real checkpoint runs are slow and memory-intensive; CUDA uses
bfloat16, MPS uses float16, and CPU defaults to float32. Normal unit tests use
the tiny local decoder and require neither network access nor model weights.

This is an experimental detector, not a security boundary. Thresholds require
calibration on held-out traffic. Automatic updates accept only extremely
high-confidence positive pseudo-labels, cap their replay share, mix balanced
trusted examples, apply diagonal-Fisher EWC, create a rollback checkpoint, and
publish atomically after safeguards pass. Those controls reduce forgetting but
do not eliminate data-poisoning risk; lower-confidence flags always require
manual review and the system never invents automatic benign labels.

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
