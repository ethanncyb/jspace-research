# 01 — Requirements and scope

## Research question

Measure how inference-time changes in selected J-Space representations affect
Qwen's ability to solve GSM8K problems, while keeping the model, prompts,
dataset examples, decoding, hardware class, and evaluator fixed.

The benchmark must support:

- a normal baseline with no activation modification;
- an observation-only baseline that records J-Space details;
- an exact no-op hook control;
- one or more configured J-Space interventions;
- paired, per-example comparison between conditions.

The application and all metric logic are Python modules. Jupyter notebooks are
the supported human-facing visualization layer over saved artifacts.

## Primary metric

The primary metric is GSM8K exact-answer accuracy:

```text
accuracy = examples with normalized predicted answer == normalized gold answer
           ---------------------------------------------------------------
                              evaluated examples
```

The evaluator extracts the final numeric answer from the generated text and
compares a normalized representation. It must report extraction failures
separately rather than silently treating parser behavior as model behavior.

Secondary metrics may include:

- answer extraction rate;
- average generated tokens and latency;
- baseline-to-intervention changed outcomes;
- hidden- and J-Space perturbation magnitude;
- per-layer J-Space norm and top-k vocabulary readouts;
- optional teacher-forced next-token rank metrics from
  [`jlens/benchmark.py`](../../jlens/benchmark.py).

## Configuration requirements

One YAML file must control at least:

- model name, revision, dtype, attention implementation, and device mapping;
- backend selection (`auto`, MPS, NVIDIA CUDA, ROCm, or CPU), offload,
  compatibility mode, and linear-algebra device;
- uv host profile and small/medium/large run tier;
- visualization input runs, notebook templates, export formats, and output
  directory;
- J-Lens source, revision, file, or local path;
- test type: full-answer GSM8K or optional gold-next-token controllability;
- dataset source, configuration, split, subset, shuffle, and seed;
- prompt template and answer format;
- generation settings;
- run condition: `baseline`, `no_op`, or `intervention`;
- capture layers, token selection, recorded fields, top-k, and storage format;
- intervention method, layers, token selection, feature selection, and strength;
- output root, experiment name, run ID policy, compression, and flush interval.

## Token-selection requirements

Layer and token selection are independent.

Supported layer selectors:

- all fitted layers;
- the existing automatic late-layer band;
- an explicit list;
- inclusive range plus stride.

Supported token selectors:

- `prompt_last`: final prompt token only;
- `generated_last`: final generated token only;
- `all_generated`: every generated model token;
- `generated_stride`: every Nth generated model token;
- `word_end`: the final model token associated with each whitespace-delimited
  generated word;
- explicit absolute or generated-relative positions.

“Every word” cannot be implemented as one activation per natural-language word
without a policy because models operate on subword tokens. The plan defines
`word_end` as a deterministic post-tokenization mapping and saves both the
model token position and derived word index.

## Observation requirements

For each selected example, layer, and token position, configurable capture
fields include:

- hidden-state L2 norm;
- J-Space L2 norm;
- selected hidden or J-Space vector;
- top-k decoded J-Space tokens and logits;
- model token ID and decoded token text;
- absolute position, prompt/generated-relative position, and optional word
  index;
- condition and intervention metadata.

Full vectors are disabled by default because their storage cost is large.
Norms and top-k readouts are the default detail level.

## Reproducibility requirements

Every run saves:

- the resolved configuration;
- model, tokenizer, J-Lens, and dataset identifiers/revisions;
- package versions, git commit, hardware, dtype, and seed;
- a stable dataset selection manifest;
- one immutable run manifest;
- completion and evaluation records keyed by stable example ID.
- resolved backend, device, dtype, offload/fallback behavior, and capability
  probe results.

Runs are resumable. Resume is allowed only when the existing run manifest is
compatible with the resolved configuration.

## Safety and correctness requirements

- Baseline hooks never return modified activations.
- `no_op` computes no activation delta and must match a same-hardware baseline
  completion-for-completion under greedy decoding.
- Intervention output records the actual perturbation magnitude.
- A fitted J-Lens must match the model hidden dimension and supported layers.
- Identity projection is allowed only for plumbing tests and is visibly marked
  invalid for research conclusions.
- Captured tensors are detached immediately and moved off accelerator memory.
- Device-specific code is isolated behind platform adapters; benchmark,
  capture, and intervention semantics are identical on MPS, CUDA, and ROCm.
- A memory preflight detects configurations that cannot fit before loading the
  full model.
- uv uses separate locked environments for MPS, ROCm, CUDA, and CPU; requested
  and resolved backends must agree before generation.

## Non-goals for the first implementation

- Training or fitting a new Jacobian lens.
- Claiming that a performance change uniquely localizes “math reasoning.”
- Supporting arbitrary model architectures before the Qwen path is validated.
- Distributed multi-node execution.
- Natural-language or LLM-as-judge scoring as the primary GSM8K evaluator.
- Combining HumanEval and GSM8K outputs in the same run directory.

## Acceptance criteria

The first complete version is accepted when:

1. a five-example smoke baseline runs and evaluates from one config;
2. a resumed run does not duplicate completed examples;
3. a full baseline produces exact-answer accuracy and complete metadata;
4. observation-only capture supports explicit layers and each required token
   selector;
5. a no-op run exactly matches the same-hardware baseline;
6. an intervention run saves completions, captures, perturbation statistics,
   and paired comparison results;
7. malformed configuration fails before model or dataset loading;
8. unit tests do not require downloading the 9B model.
9. M1/MPS, NVIDIA CUDA, and Radeon 8060S/ROCm each have a documented hardware
   smoke/no-op validation gate before being marked supported.
10. the M1 small run promotes to a Radeon medium run, then to an A100/H100 full
    run only after the previous tier's correctness gates pass.
11. parameterized Jupyter notebooks can regenerate readable tables, plots,
    executed notebooks, and HTML from saved runs without loading the model.
