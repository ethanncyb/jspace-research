# 04 — Data and output contracts

## Output directory layout

Each run is self-contained and immutable except while actively running:

```text
outputs/gsm8k/
└── <run_id>/
    ├── manifest.json
    ├── resolved_config.yaml
    ├── environment.json
    ├── dataset_selection.jsonl
    ├── progress.json
    ├── completions.jsonl
    ├── captures/
    │   ├── index.jsonl
    │   └── <example_id>.jsonl.gz
    ├── intervention/
    │   ├── summary.json
    │   └── hook_log.jsonl.gz
    ├── evaluation/
    │   ├── results.jsonl
    │   ├── summary.json
    │   └── report.md
    ├── visualization/
    │   ├── visualization_manifest.json
    │   ├── executed/
    │   │   └── *.ipynb
    │   ├── html/
    │   │   └── *.html
    │   ├── figures/
    │   │   ├── *.png
    │   │   └── *.svg
    │   └── tables/
    │       └── *.csv
    └── comparisons/
        └── <other_run_id>/
            ├── paired_results.csv
            ├── summary.json
            └── report.md
```

The run ID should be human-readable plus collision-resistant, for example:

```text
baseline_qwen35-9b_7fd29c1a_20260819T230000Z
```

Visualization artifacts are derived outputs. Their manifest records notebook
template, source/input checksums, uv lock checksum, and analysis package
version. Regeneration never modifies completions, captures, or evaluation
results.

## Manifest contract

`manifest.json` is written before generation begins and finalized when the run
ends:

```json
{
  "schema_version": 1,
  "run_id": "baseline_qwen35-9b_7fd29c1a_20260819T230000Z",
  "status": "running",
  "condition": "baseline",
  "test_type": "full_answer",
  "config_fingerprint": "sha256:...",
  "dataset_selection_fingerprint": "sha256:...",
  "model": {
    "name": "Qwen/Qwen3.5-9B-Base",
    "revision": null,
    "dtype": "bfloat16"
  },
  "jlens": {
    "status": "fitted",
    "source": "neuronpedia/jacobian-lens",
    "revision": "qwen-n1000",
    "supported_layers": [0, 1, 2]
  },
  "resolved_capture_layers": [10, 11, 12],
  "resolved_intervention_layers": [],
  "started_at": "2026-08-19T23:00:00Z",
  "finished_at": null,
  "completed_examples": 0
}
```

On success, `status` becomes `complete`. Interrupted runs remain `running` or
are marked `interrupted` on the next resume.

## Dataset selection record

`dataset_selection.jsonl` fixes the evaluated population before model
generation:

```json
{
  "example_id": "gsm8k_test_000042",
  "source_index": 42,
  "dataset": "openai/gsm8k",
  "dataset_config": "main",
  "split": "test",
  "question_sha256": "...",
  "gold_answer": "18"
}
```

The full gold rationale may be omitted from routine artifacts to keep the
selection file compact, but the source index and dataset revision must make it
reconstructable.

## Completion record

One JSON object per example:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "example_id": "gsm8k_test_000042",
  "source_index": 42,
  "model": "Qwen/Qwen3.5-9B-Base",
  "condition": "baseline",
  "prompt_template": "zero_shot_cot_v1",
  "prompt": "Question: ...\nAnswer: ...",
  "generated_text": "... Therefore, the answer is #### 18",
  "generated_token_ids": [123, 456],
  "n_prompt_tokens": 57,
  "n_generated_tokens": 83,
  "finish_reason": "eos",
  "seed": 0,
  "elapsed_seconds": 4.21,
  "capture_file": "captures/gsm8k_test_000042.jsonl.gz",
  "timestamp": "2026-08-19T23:00:05Z"
}
```

Records are appended only after generation and capture writing for that
example succeed. This makes `completions.jsonl` the authoritative resume log.

## Capture record

Default norm/top-k row:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "example_id": "gsm8k_test_000042",
  "condition": "baseline",
  "layer": 18,
  "phase": "pre_intervention",
  "forward_index": 12,
  "absolute_position": 68,
  "generated_position": 11,
  "word_index": 6,
  "token_id": 220,
  "token_text": " 18",
  "hidden_norm": 44.281,
  "jspace_norm": 38.902,
  "top_jspace_tokens": [
    {"token_id": 220, "text": " 18", "logit": 16.42}
  ],
  "intervention_delta_jspace_norm": 0.0,
  "intervention_delta_hidden_norm": 0.0
}
```

If full vectors are requested, do not inline thousands of float values into
JSON. Prefer Parquet fixed-size list columns or per-example tensor files with
the JSON/Parquet row storing a vector offset/reference.

## Capture index

`captures/index.jsonl` has one row per example and allows integrity checks
without opening every compressed file:

```json
{
  "example_id": "gsm8k_test_000042",
  "path": "captures/gsm8k_test_000042.jsonl.gz",
  "rows": 1411,
  "layers": [10, 14, 18, 22, 26],
  "token_mode": "all_generated",
  "sha256": "..."
}
```

## Evaluation result

```json
{
  "schema_version": 1,
  "run_id": "...",
  "example_id": "gsm8k_test_000042",
  "gold_answer_raw": "18",
  "gold_answer_normalized": "18",
  "predicted_answer_raw": "18",
  "predicted_answer_normalized": "18",
  "extraction_method": "answer_marker",
  "extraction_succeeded": true,
  "correct": true
}
```

`evaluation/summary.json` includes:

- `n_selected`, `n_completed`, and `n_evaluated`;
- `n_correct` and `accuracy`;
- extraction successes/failures and extraction rate;
- finish-reason counts;
- mean/median generated token count;
- condition and fingerprints.

## Environment record

`environment.json` makes runs from different machines auditable. It records:

- OS, architecture, Python, PyTorch, Transformers, Accelerate, and jlens
  versions;
- resolved backend: `mps`, `cuda`, `rocm`, or `cpu`;
- device names/count and available memory where the backend exposes it;
- CUDA build/runtime or HIP/ROCm version;
- MPS build/availability on Apple Silicon;
- requested/resolved dtype and device map;
- model/J-Lens offload and pseudo-inverse compute/cache devices;
- strict/compatible fallback mode and capability-probe results.
- uv version, lock-file checksum, selected virtual environment, requested torch
  backend, host profile, and small/medium/large run tier.

Paired comparison warns when these differ. The primary baseline/intervention
comparison should use the same backend because floating-point generation can
diverge across MPS, CUDA, and ROCm even with greedy decoding.

## Paired comparison

Comparison joins by `example_id`, never by row order. Each paired row contains:

- baseline and candidate correctness;
- baseline and candidate normalized predictions;
- change type: `correct_both`, `incorrect_both`, `broken`, or `fixed`;
- generated token counts;
- intervention perturbation aggregates where available.

The summary reports:

- both accuracies and absolute/relative change;
- counts for all four paired outcome categories;
- McNemar's exact test on `broken` versus `fixed` when the sample is large
  enough to be meaningful;
- bootstrap confidence interval for paired accuracy difference;
- compatibility checks and any excluded/missing examples.

## Answer normalization contract

The versioned `gsm8k_numeric_v1` parser:

1. Prefer text following the final configured `####` marker.
2. Otherwise optionally use the last numeric expression in the completion.
3. Strip commas, currency symbols, surrounding whitespace, and terminal
   punctuation.
4. Parse integers, decimals, signs, and simple fractions exactly using decimal
   or rational arithmetic.
5. Compare canonical numeric values with configured tolerance.
6. Record extraction failure when no supported numeric answer exists.

Parser unit tests must cover negative values, commas, decimals, fractions,
multiple intermediate numbers, malformed markers, empty output, and scientific
notation policy.

## Storage estimates and defaults

Norms/top-k at many layers and every generated token can still produce millions
of rows over 1,319 GSM8K test examples. Therefore:

- gzip-compressed JSONL is the dependency-light default;
- Parquet is recommended for analysis-heavy full runs;
- full vectors default to off;
- writing is streamed per example;
- capture files are sharded by example for safe resume and partial analysis;
- reports read captures as streams rather than loading the corpus into memory.

## Artifact compatibility

Every schema has `schema_version`. Readers reject unknown major versions and
must not infer missing research settings from folder names. Comparison checks
the manifests before reading result rows.
