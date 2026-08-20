# 03 — Configuration design

## Configuration principles

- YAML is the user-facing format.
- Configuration is parsed into typed structures and validated once.
- The resolved configuration, including defaults, is saved in every run.
- CLI options are limited to operational overrides such as `--limit`,
  `--run-id`, and `--resume`; research parameters remain visible in YAML.
- Unknown keys are errors to catch spelling mistakes.

## Proposed complete configuration

```yaml
schema_version: 1

experiment:
  name: "gsm8k-qwen35-9b"
  condition: "baseline"        # baseline | no_op | intervention
  run_tier: "small"            # small | medium | large
  tags: ["gsm8k", "jspace"]
  notes: null

model:
  name: "Qwen/Qwen3.5-9B-Base"
  revision: null
  tokenizer_revision: null
  dtype: "auto"                # auto | bfloat16 | float16 | float32
  device_map: "auto"
  offload_folder: null
  attention_implementation: null
  trust_remote_code: false

jlens:
  required: true
  source: "hub"                # hub | local | identity
  repo: "neuronpedia/jacobian-lens"
  revision: "qwen-n1000"
  file: "qwen3.5-9b-pt/jlens/Salesforce-wikitext/Qwen3.5-9B-Base_jacobian_lens.pt"
  local_path: null

benchmark:
  test_type: "full_answer"     # full_answer | gold_next_token
  dataset: "openai/gsm8k"
  dataset_config: "main"
  split: "test"
  full_run: false
  subset_size: 10
  selection: "first"           # first | shuffled
  selection_seed: 0

prompt:
  template: "zero_shot_cot_v1" # provisional until prompt pilot is frozen
  answer_marker: "####"
  few_shot_examples: 0
  context_overflow: "error"     # error | reduce_few_shot
  final_token_capture: "replay" # replay | predicting_state

generation:
  mode: "greedy"               # greedy | sample
  max_new_tokens: 512
  do_sample: false
  temperature: 0.0
  top_p: 1.0
  seed: 0
  stop_strings: []

capture:
  enabled: true
  phases: ["post_layer"]        # initially only post_layer
  layers:
    mode: "late"                # late | all_fitted | explicit | range
    values: []
    start: null
    stop: null
    stride: 1
  tokens:
    mode: "all_generated"       # prompt_last | generated_last |
                                # all_generated | generated_stride |
                                # word_end | explicit
    stride: 1
    positions: []
    include_prompt: false
  fields:
    hidden_norm: true
    jspace_norm: true
    top_jspace_tokens: true
    hidden_vector: false
    jspace_vector: false
    intervention_delta_norm: true
  top_k_tokens: 20
  vector_dtype: "float16"
  capture_pre_intervention: true
  capture_post_intervention: false

intervention:
  enabled: false
  method: "mean_replace"        # mean_replace initially
  layers:
    mode: "late"
    values: []
    start: null
    stop: null
    stride: 1
  tokens:
    mode: "all_generated"
    stride: 1
    positions: []
    include_prompt: true
  features:
    mode: "top_abs"             # top_abs | explicit | random_matched
    top_k: 50
    indices: []
    random_seed: 0
  strength: 0.05
  reference: "task_running_mean"

evaluation:
  parser: "gsm8k_numeric_v1"
  prefer_answer_marker: true
  allow_last_number_fallback: true
  numeric_tolerance: 0.0

outputs:
  root_dir: "outputs/gsm8k"
  run_id: null                  # label only; UTC stamp is appended to the folder
  on_existing: "error"          # error | resume
  completion_format: "jsonl"
  capture_format: "jsonl_gzip"  # jsonl_gzip | parquet
  save_prompts: true
  save_generated_token_ids: true
  flush_every_examples: 1

visualization:
  enabled: true
  notebooks:
    - "run-overview"
    - "gsm8k-accuracy"
  export_formats: ["ipynb", "html"]
  figure_formats: ["png", "svg"]
  save_backing_tables: true
  max_examples_in_tables: 100
  max_capture_rows_in_memory: 250000

runtime:
  host_profile: "auto"         # auto | m1-max | radeon-8060s |
                                # nvidia-datacenter | cpu
  backend: "auto"              # auto | mlx | mps | cuda | rocm | cpu
  device: "auto"
  linear_algebra_device: "auto" # auto | model | cpu
  compatibility_mode: "strict" # strict | compatible
  memory_preflight: true
  minimum_free_memory_gb: 2.0
  pinv:
    compute_device: "cpu"
    cache_device: "cpu"
    cache_dir: ".cache/jspace/pinv"
    preload: false
  offline: false
  deterministic_algorithms: false
  log_level: "INFO"
```

## Cross-platform runtime resolution

The backend resolver must distinguish:

- Apple Silicon through MPS;
- NVIDIA through CUDA;
- AMD Radeon 8060S through ROCm/HIP;
- CPU and explicit CPU/offload paths.

ROCm uses much of PyTorch's `torch.cuda` API, so `torch.version.hip` is checked
before treating an available accelerator as NVIDIA CUDA.

`dtype: auto` resolves to a capability-tested dtype: bfloat16 where supported
on CUDA/ROCm, float16 on MPS, and float32 on CPU. The resolved value is saved
and never silently changes during a run.

`linear_algebra_device` is separate because pseudo-inverse support and memory
cost differ from model inference. The portable default computes/caches fitted
layer pseudo-inverses in float32 on CPU and transfers only required values.

The memory preflight estimates model weights, J-Lens matrices, configured
pseudo-inverses, KV cache, and capture buffers. It gives an actionable failure
before attempting the 9B load on an M1 or accelerator that cannot fit the
selected experiment.

`host_profile` controls only execution mechanics. `run_tier` labels the
intended small/medium/large protocol, but it does not silently change dataset,
capture, or intervention settings. Those values come from the resolved YAML and
are fingerprinted normally.

## Condition validation

The `condition` field is authoritative:

| condition | capture allowed | intervention.enabled | behavior |
|---|---:|---:|---|
| `baseline` | yes | must be false | no modifying hooks |
| `no_op` | yes | must be false | hooks installed, exact zero delta |
| `intervention` | yes | must be true | configured activation change |

The validator rejects contradictory states instead of silently changing them.

## Layer selector resolution

The same selector schema is used independently for capture and intervention.

- `late`: use the established J-Lens late band intersected with fitted layers.
- `all_fitted`: all layers present in the fitted lens.
- `explicit`: exactly `values`.
- `range`: Python-style `[start, stop)` with `stride`.

Resolved layers are sorted, deduplicated, checked against model layers and
J-Lens support, then written into `resolved_config.yaml`.

Example:

```yaml
capture:
  layers:
    mode: "explicit"
    values: [10, 14, 18, 22, 26]
```

## Token selector behavior

Token selectors operate on generation events, not decoded strings inside model
hooks.

- `prompt_last`: captures the prefill activation at the last prompt position.
- `all_generated`: captures each decode position.
- `generated_stride`: captures generated positions `0, N, 2N, ...`.
- `generated_last`: requires a small final-state handoff because the final
  token is only known after generation stops.
- `word_end`: captures token rows during generation, then marks/retains rows
  whose decoded character span ends a whitespace-delimited word.
- `explicit`: uses generated-relative positions unless `include_prompt` is
  true and an explicit prompt-relative convention is supplied.

Example for every generated word:

```yaml
capture:
  tokens:
    mode: "word_end"
    include_prompt: false
```

Example for only the final answer token:

```yaml
capture:
  tokens:
    mode: "generated_last"
    include_prompt: false
```

## Test types

### `full_answer`

Generate a complete rationale/answer and report GSM8K exact-answer accuracy.
This is the primary experiment.

### `gold_next_token`

Use deterministic teacher-forced reference prefixes and report token rank,
logit lift, and KL metrics. This delegates to or adapts
[`jlens/benchmark.py`](../../jlens/benchmark.py). Output must carry
`metric_scope: controllability` and must never be labeled GSM8K accuracy.

## Output override examples

Separate experiments:

```yaml
outputs:
  root_dir: "/mnt/experiments/gsm8k"
  run_id: "qwen9b-baseline-seed0"
```

Compact final-token capture:

```yaml
capture:
  tokens:
    mode: "generated_last"
  fields:
    hidden_norm: true
    jspace_norm: true
    top_jspace_tokens: true
    hidden_vector: false
    jspace_vector: false
```

Full vectors at selected layers and every fifth generated token:

```yaml
capture:
  layers:
    mode: "explicit"
    values: [10, 18, 26]
  tokens:
    mode: "generated_stride"
    stride: 5
  fields:
    hidden_norm: true
    jspace_norm: true
    top_jspace_tokens: false
    hidden_vector: true
    jspace_vector: true
```

## Configuration fingerprints

Two fingerprints should be saved:

- `run_fingerprint`: all result-affecting settings, dataset selection, model,
  tokenizer, and lens revisions;
- `condition_fingerprint`: intervention/capture condition only.

Comparison requires matching benchmark, selection, prompt, generation, model,
and lens fields. It permits expected differences in condition and output
settings.
