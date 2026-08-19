# 06 — Implementation and validation

## Milestone 1 — Package and typed configuration

Implement:

- package/CLI skeleton;
- uv project metadata, committed lock, and isolated backend environments;
- host detection/verification launcher for MPS, ROCm, CUDA, and CPU;
- typed configuration models;
- YAML loading, defaults, unknown-key rejection, and cross-field validation;
- resolved config and fingerprints;
- smoke/default/condition config files.

Tests:

- host-profile mapping and explicit override behavior;
- requested/resolved torch backend mismatch failure;
- valid baseline, no-op, and intervention configs;
- invalid condition combinations;
- layer/token selector validation;
- unknown key and unsupported method failures;
- stable fingerprint behavior.

Completion gate: `uv sync --locked` creates the selected isolated environment,
the verifier identifies its backend correctly, and `inspect-config` prints a
complete resolved config without loading the full model.

## Milestone 2 — GSM8K data, prompts, and evaluator

Implement:

- deterministic GSM8K loading and selection manifest;
- stable IDs;
- versioned zero-shot chain-of-thought prompt;
- versioned numeric answer parser;
- evaluation and summary writers.

Tests:

- deterministic first/shuffled subsets;
- calculator annotation cleaning;
- stable IDs across reloads;
- answer parser edge cases;
- exact and tolerance-based comparison;
- reevaluation solely from saved completions.

Completion gate: fixture completions produce expected exact-answer accuracy.

## Milestone 3 — Model and J-Lens adapters

Implement:

- model/tokenizer loading;
- MPS, NVIDIA CUDA, ROCm/HIP, and CPU backend detection;
- capability-tested dtype/device resolution;
- memory preflight and environment diagnostics;
- separate model and linear-algebra device policies;
- `jlens.from_hf` wrapping;
- fitted/hub/local/identity J-Lens loading;
- dimension and layer compatibility checks;
- normalized adapter metadata.

Reuse the proven concepts from
[`qwen-humaneval-jlens/src/load_model.py`](../../qwen-humaneval-jlens/src/load_model.py)
and
[`qwen-humaneval-jlens/src/load_jlens.py`](../../qwen-humaneval-jlens/src/load_jlens.py),
without preserving HumanEval phase naming.

Tests:

- fake model/lens compatibility;
- mismatched hidden dimension;
- unsupported explicit layer;
- identity lens marked as non-research placeholder.
- ROCm is not mislabeled as NVIDIA CUDA;
- MPS resolves to float16 unless a requested dtype probe passes;
- unsupported operations fail in strict mode;
- memory preflight rejects an intentionally infeasible 9B configuration;
- CPU pseudo-inverse plus accelerator back-projection preserves shapes/devices.

Completion gate: a tiny local model can generate through the common adapter on
CPU and each available accelerator without benchmark-specific device branches.

## Milestone 4 — Artifacts, unified runner, and resume

Implement:

- run directory and manifest lifecycle;
- immutable dataset selection;
- one shared generation function;
- atomic/append-safe completion writing;
- completion-based resume;
- manifest compatibility checks;
- interruption status and progress.

Tests:

- baseline run with fake components;
- interrupted run resumes without duplicates;
- changed result-affecting config cannot resume;
- a partial capture without a completion is safely rerun;
- duplicate example IDs fail.

Completion gate: five synthetic examples survive forced interruption/resume.

## Milestone 5 — Observation-only capture

Implement:

- layer resolver;
- step/position context;
- `prompt_last`, `all_generated`, `generated_stride`, `generated_last`,
  `word_end`, and explicit selectors;
- norm and top-k capture;
- optional vector capture;
- per-example streaming storage and index;
- hook cleanup on normal completion and exceptions.

Use
[`qwen-humaneval-jlens/src/capture_jspace.py`](../../qwen-humaneval-jlens/src/capture_jspace.py)
as a behavioral reference, but correct its design limitation: it always
observes only the final position of each forward call and cannot directly
express all requested selector policies.

Tests:

- hook never returns modified output;
- call/absolute/generated positions align;
- token IDs align after generation;
- all selector modes;
- capture disabled installs no hooks;
- top-k and vector settings;
- no retained accelerator tensors.

Completion gate: observation-only tiny-model completions exactly match capture
disabled completions.

## Milestone 6 — No-op and intervention controllers

Implement:

- common controller interface;
- exact no-op controller;
- `mean_replace` port;
- task-local running state reset;
- pre/post capture phases;
- perturbation summaries;
- future extension point for `random_matched`, `zero_topk`, and other controls.

Important implementation constraint:

```text
delta_z = transformed_z - original_z
delta_h = pinv(J_layer) @ delta_z
output = original_hidden + strength-controlled delta_h
```

For no-op, skip projection-back and return the original output object exactly,
not a cloned equivalent. This provides the strongest bitwise control.

Tests:

- no-op output object/value behavior;
- task reset prevents state leakage;
- strength zero is exact;
- invalid strength/top-k;
- selected layers/tokens only;
- deterministic random-matched selection;
- tuple and tensor layer output handling;
- perturbation statistics use original activations as denominators.

Completion gate: same-hardware no-op smoke matches baseline generated token IDs
for every example.

## Milestone 7 — Comparison and reports

Implement:

- manifest compatibility checker;
- paired example join;
- changed-outcome classification;
- paired accuracy statistics;
- capture/intervention aggregate summaries;
- machine-readable summary plus Markdown report.

Tests:

- missing and extra examples;
- incompatible prompt/model/dataset selections;
- all four outcome categories;
- zero baseline accuracy;
- confidence interval reproducibility;
- parser failures represented separately.

Completion gate: report can be regenerated from completed run directories
without loading model or dataset.

## Milestone 8 — Python analysis and Jupyter notebooks

Implement:

- schema-aware, streaming artifact loaders;
- reusable table, statistics, and plot functions;
- parameterized overview, accuracy, capture, intervention, hardware, and
  example-explorer notebooks;
- uv `analysis` dependency group;
- automated notebook execution and HTML/figure/table export;
- visualization manifests with source and input checksums.

Tests:

- analysis aggregates match evaluator/comparison summaries;
- large capture filters apply before pandas materialization;
- every notebook has a tagged parameter cell;
- every notebook executes against tiny fixture runs;
- cell errors fail automated export;
- HTML, PNG/SVG, CSV, and executed notebook outputs are saved.

Completion gate: all notebooks execute from saved fixture artifacts without
loading a model or requiring a GPU.

## Milestone 9 — End-to-end experiment validation

Execute in order:

1. M1 Max: 5–10-example baseline/capture/no-op/intervention smoke;
2. Radeon 8060S: fixed 100-example real-model validation and calibration;
3. A100 or H100: full 1,319-example baseline and no-op gate;
4. A100 or H100: full observation baseline;
5. A100 or H100: frozen intervention and matched controls;
6. paired reports and separate cross-hardware replication summary.

Do not begin the next stage if the preceding correctness gate fails.
Do not pair a baseline from one hardware class with an intervention from
another.

## Test strategy

### Fast unit tests

Run on CPU with fake tokenizers, toy layers, and synthetic datasets. These cover
most logic and are required in normal CI.

### Integration tests

Use the repository's tiny decoder pattern from
[`tests/tiny.py`](../../tests/tiny.py) or a small Hugging Face causal model.
Network-dependent tests are marked and excluded by default.

### Hardware smoke tests

Run separately on M1/MPS, NVIDIA CUDA, and Radeon 8060S/ROCm. Verify
backend labeling, dtype/device placement, hooks, pseudo-inverse policy, no-op
identity, and memory stability. A backend is not marked validated based only on
CPU tests.

### Full benchmark tests

These are experiment runs, not CI tests. Their manifests and reports are the
validation evidence.

## Performance considerations

- Cache each layer pseudo-inverse once, but estimate memory before eagerly
  caching many 4,096 × 4,096 float32 matrices.
- Consider computing/caching only intervention layers.
- Default pseudo-inverse computation and caching to CPU for portable behavior;
  permit MPS/CUDA/ROCm only after an operation probe.
- Detach and transfer capture values to CPU immediately.
- Batch size one is the initial correctness path because hooks and task-local
  state are per example.
- Stream capture rows and compress per example.
- Avoid computing top-k vocabulary readouts when disabled; unembedding is
  expensive.
- Resolve `generated_last` without retaining all full hidden vectors.

## Risks and mitigations

| risk | mitigation |
|---|---|
| capture changes output | read-only hook test and clean-vs-observe hash check |
| no-op changes output | return original output object and require bitwise gate |
| subword/word ambiguity | versioned `word_end` mapping with saved word index |
| huge capture volume | default norms/top-k only, sharding, compression, estimates |
| model/J-Lens mismatch | strict dimension/layer/revision metadata validation |
| config drift | resolved config, fingerprints, immutable run manifest |
| parser inflates/deflates accuracy | versioned deterministic parser and extraction report |
| intervention destroys fluency | calibration sweep and perturbation/length checks |
| task state leaks | explicit per-example reset plus tests |
| resume mixes experiments | manifest compatibility check |
| ROCm mistaken for CUDA | inspect `torch.version.hip` before CUDA labeling |
| M1 cannot fit 9B experiment | preflight, explicit offload, or matching smaller model |
| backend lacks linear algebra op | CPU pseudo-inverse policy and strict capability probe |
| cross-device numeric divergence | same-backend primary comparison and environment manifest |

## Definition of done

- The project tree and configs are documented in its top-level README.
- Every user-controlled requirement is represented in the typed config.
- Baseline/no-op/intervention use one generation path.
- Full-answer exact accuracy is clearly separated from next-token
  controllability.
- Capture supports configurable layers and required token policies.
- Saved details can be analyzed without reloading the model.
- Runs are reproducible, resumable, and comparison-safe.
- Fast tests pass locally; MPS, CUDA, and ROCm hardware smoke/no-op gates are
  recorded before those backends are marked validated.
- A full baseline and at least one controlled intervention produce complete
  paired reports.
