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

## Promptguard: evolving activation-level injection detector

The `promptguard` package is a research prototype for detecting prompt
injection and jailbreak drift in Qwen 3.5 residual streams. It observes the
eight full-attention checkpoint blocks in the 4B model, fits a layerwise linear
probe to paired activation deltas, and can either remove the learned drift
direction during generation or stop with a fixed refusal. Qwen's Gated
DeltaNet blocks remain read-only.

This is an experimental detector, not a production security boundary. Keep
the model in an isolated environment, use versioned human-reviewed datasets,
and do not treat self-generated labels as ground truth.

### Setup and first POC run

Install the package and dependencies, then edit [`config.yaml`](config.yaml).
The bundled prompt lists are deliberately tiny bootstrap data; replace them
before drawing conclusions. The default configuration runs ten evolution
rounds. Qwen3.5-4B is roughly a 9.3 GB download and needs substantially more
memory while running.

```bash
uv sync
uv run python evolution_loop.py --config config.yaml
```

Override all evolution randomness, including active-prompt order, while keeping
the benchmark split fixed with `--seed`. Use per-seed output directories when
running replicates:

```bash
uv run python evolution_loop.py --config config.yaml --seed 19 \
  --output-dir outputs/evolution_multiseed/seed_19 \
  --checkpoint-dir outputs/checkpoints_multiseed/seed_19
```

Aggregate completed replicates into a mean ± sample-standard-deviation CSV and
chart with:

```bash
uv run python aggregate_seeds.py outputs/evolution_multiseed/seed_{7,19,31} \
  --output-dir outputs/evolution_multiseed
```

Inspect semantically encoded attacks outside the evolution loop with a saved
probe checkpoint. This writes one row per prompt/layer, a family/layer summary,
and a chart comparing probe alignment with raw activation-delta magnitude:

```bash
uv run python layer_diagnostic.py --config config.yaml \
  --probe outputs/checkpoints_multiseed/seed_7/guard_000.pt \
  --output-dir outputs/layer_diagnostic/guard_000
```

The command performs the following sequence:

1. load `Qwen/Qwen3.5-4B`, discovering full-attention blocks from
   `text_config.layer_types` (the configured indices are a fallback);
2. train `guard_000.pt` on the raw trusted benign/malicious seed set;
3. transform malicious seeds with the evolving strategy pool and update its
   row-normalized Markov/Q transition matrix after failed attacks;
4. continue training the *same* probe weights on successful attacks, matched
   benign disguises, and raw examples weighted by `lambda_raw`;
5. save each `guard_NNN.pt` and strategy-pool snapshot, then evaluate every
   Attack-K/Guard-J pair plus an untrained held-out family.

Attacker reflection is routed through Olares and returns a `MutationProposal`
template, never executable Python. Mutations are embedding-deduplicated and
enter the active pool only when their ASR against the frozen validation probe
reaches `attacker.sandbox_tau`.

Low-confidence pseudo-labels are not trained on. `ConfidenceGate` appends them
to `outputs/review_queue.csv`; only values at or above
`guard.confidence_gate` are admitted automatically. Trusted attack seeds and
matched benign seeds bypass this pseudo-label gate because their labels are
provided by the dataset.

### Olares configuration and fallback behavior

Copy [`.env.example`](.env.example), set the endpoint/key, and export it into
the shell before running. The API key is read from the environment variable
named by `olares.api_key_env`; it is never written to result CSVs.

```bash
cp .env.example .env
# edit .env, then:
set -a
source .env
set +a
```

At startup the client probes `/v1/chat/completions` with a minimal chat message
and a 10-second minimum timeout. Text generation uses only that endpoint with
the configured bearer token. Every later request also catches timeouts, network
errors, and malformed responses independently. If Olares fails mid-run, text
generation falls back to `Qwen/Qwen2.5-1.5B-Instruct` and embeddings fall back
to `sentence-transformers/all-MiniLM-L6-v2`; failure of the embedding fallback
uses a stable local hash so strategy deduplication cannot crash the run. Set
`olares.force_local_fallback: true` to test that path deliberately.

Model routing is config-driven:

- `fast_model` (`Ornith-1.0-9B` by default): reflection, ReNeLLM rewrites, and
  TAP candidates;
- `reason_model` (`glm-fixed`): optional harmfulness judge only;
- `embed_model` (`nomic-embed-text`): strategy similarity/deduplication;
- `code_model` (`qwen3-coder:30b`): reserved for future code strategies.

No paid API is required. Olares or the local Hugging Face fallbacks cover every
LLM call in the default path.

### HarmBench/AdvBench benchmark

The loader downloads and caches the official
[HarmBench behavior CSV](https://github.com/centerforaisafety/HarmBench/tree/main/data/behavior_datasets)
and [AdvBench harmful-behavior CSV](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv).
Defaults select 25 instructions from each dataset (50 total) and reserve 20%
(preferably whole semantic categories) for held-out evaluation. Use `full` for
either subset size later; no code change is needed. The default evolution run
uses this benchmark split in place of the tiny smoke-test seeds, while keeping
held-out categories/instructions out of guard updates.

After an evolution run, compare the configured baseline attack and evolving
strategy snapshots across every saved guard:

```bash
uv run python benchmark_runner.py --config config.yaml
```

The default baseline is one ReNeLLM-style rewrite. Add `tap` to
`baselines.enabled` to enable iterative candidate generation/pruning within
`baselines.query_budget`. The optional Olares reason-model judge is disabled by
default; enable `metrics.enable_harmfulness_judge` only when response-quality
scoring is desired.

Benchmark outputs under `outputs/benchmarks/` are:

- `asr_aqc.csv`: DARWIN Table-1/4-style ASR and average query count (AQC), per
  dataset and aggregated, with guard round, strategy-pool size, and backend;
- `attempts.csv`: per-instruction best score, attempts, split, and prompt;
- `cross_stage.csv`: small Attack-K × Guard-J matrix for forgetting and
  strategy-pool scaling.

Here, **ASR means the harmful prompt stayed below this internal drift probe's
threshold**, and AQC counts probe attempts until first evasion or budget
exhaustion. These are not attacks against a frontier model and, by default, do
not use GPT-4 or HarmBench's response classifier. They are therefore not
directly comparable to DARWIN's published ASR/AQC numbers.

### Evaluate hooks, probe, and intervention

After a round, score the configured datasets without generation:

```bash
uv run python -m promptguard.eval_harness \
  --config config.yaml \
  --probe outputs/checkpoints/guard_003.pt
```

Add `--generate` to run the configured hard-stop or circuit-breaker behavior
and include generated continuations in the detailed CSV. Omitting it is useful
for fast detector-only sweeps; the intervention trigger decision is still
reported.

`outputs/eval/examples.csv` contains one row per example. `drift_score` is the
aggregate unsafe probability, `layer_<n>_score` shows where drift becomes
linearly separable, and `intervention_triggered` records policy activation.
`outputs/eval/summary.csv` reports AUC, accuracy, false-positive and
false-negative rates, unsafe recall, benign pass rate, and trigger rate.

Evolution artifacts are plot-ready CSVs:

- `round_metrics.csv`: pool size, attacker ASR, unsafe recall, and benign pass
  rate by round (the Table 4 / Figure 3-style overview);
- `active_by_round.csv`: active attacker ASR by sampled strategy family and
  overall active-payload ASR;
- `heldout_by_round.csv`: exhaustive held-out-payload ASR by attack family,
  including dedicated translated and acrostic/encoding categories plus
  roleplay, context, fragmentation, structured-data, and other families;
- `benchmark_split.csv`: the reproducible HarmBench/AdvBench active/held-out
  assignment and source semantic category;
- `attacks.csv`: each retry, strategy family, probe score, and success;
- `cross_stage.csv`: Attack-K × Guard-J ASR. Values that rise for old Attack-K
  as J increases indicate forgetting; falling held-out ASR suggests transfer
  to a strategy family never used for training.

### Visualizations

`visualize.py` is a pure plotting layer: it reads the CSV artifacts above and
the benchmark/evaluation CSVs, and does not load Qwen, rerun prompts, or collect
new experiment data. Generate the complete presentation-ready suite under
`outputs/charts/` with:

```bash
uv run python visualize.py --all
```

The suite uses a consistent visual language (benign/usability metrics in
blue-green and attacks/unsafe metrics in red-orange) and includes:

- `layer_drift_lines.png`: overlaid benign and attack trajectories across the
  eight hooked full-attention layers, including bold group means;
- `layer_drift_heatmap.png`: prompt-by-layer drift intensity, ordered to expose
  clusters of attack-like activation patterns;
- `asr_over_rounds.png`: evolving-attacker, ReNeLLM, and TAP ASR by guard round
  (only methods present in the benchmark CSV are drawn);
- `guard_robustness.png`: unsafe recall and benign pass rate together, making
  detection/usability tradeoffs visible;
- `cross_stage_matrix.png`: DARWIN-style Attack-K × Guard-J ASR cross-play for
  spotting forgetting and cross-round generalization;
- `strategy_pool_composition.png`: normalized strategy-selection frequency by
  evolution round, derived from `attacks.csv`;
- `intervention_triggers.png`: layer trajectories, policy threshold, and first
  recorded threshold crossing for circuit-breaker and/or hard-stop samples;
- `heldout_generalization.png`: ASR of the never-trained held-out attack family
  against each guard checkpoint.

Regenerate one chart with `--chart`, for example:

```bash
uv run python visualize.py --chart layer_drift_heatmap
uv run python visualize.py --chart intervention_triggers --threshold 0.50
```

Use `python visualize.py --help` for CSV-path and output-directory overrides.
The intervention chart infers the first crossing from existing per-layer CSV
scores because evaluation records the policy decision but not a separate
trigger-layer field. If eval data contains both intervention modes, they are
shown side by side; otherwise the available triggered mode is shown alone.

To swap to Qwen 3.5 9B or another compatible decoder, change `model.name` and
`model.hidden_dim`. Full-attention indices are automatically inferred when the
checkpoint publishes `layer_types`; otherwise set `model.layer_indices` and,
for unusual Hugging Face wrappers, `model.layer_path`.

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
