# 05 — Experiment protocol

## Experimental conditions

### A. Clean baseline

- No modifying forward hooks.
- Greedy generation using the selected prompt template.
- Observation capture may be disabled for the pure performance baseline.
- Establishes GSM8K exact-answer accuracy and generation behavior.

### B. Observation-only baseline

- Same dataset selection, model, prompt, and generation settings as A.
- Read-only capture hooks enabled at configured layers and token positions.
- Used to verify capture overhead and collect normal J-Space trajectories.
- Must have no systematic accuracy or completion differences from A.

### C. No-op hook control

- Same hook registration points and output reconstruction path as intervention.
- The J-Space delta is exactly zero.
- Must reproduce clean baseline completions bit-for-bit on the same hardware
  under greedy decoding.
- This gate must pass before condition D is interpreted.

### D. J-Space intervention

- Same shared runner and generation path.
- Only configured condition/controller settings differ.
- Capture records pre-intervention states and actual perturbation magnitudes.

## Frozen variables

For a valid paired comparison, these must match:

- model and tokenizer repository/revision;
- fitted J-Lens checkpoint;
- dataset revision and selected example IDs;
- prompt template/version and few-shot examples;
- decoding parameters and seed;
- max input/output token policy;
- answer parser version;
- hardware class and software environment where practical.

Output location, capture storage format, and run ID may differ.

## Phase 0 — Parser and plumbing validation

1. Unit-test answer extraction against hand-written cases.
2. Run the pipeline with tiny/fake model components.
3. Verify resolved config and run manifest generation.
4. Interrupt and resume a synthetic run.
5. Verify malformed config fails before loading external artifacts.
6. Run backend, dtype, operation, and memory preflight diagnostics.

Exit gate: all fast tests pass without downloading Qwen.

## Platform qualification

Before a backend is used for a full experiment:

1. identify it correctly as MPS, NVIDIA CUDA, ROCm/HIP, or CPU;
2. pass matrix multiplication, norm, top-k, unembedding, and configured
   pseudo-inverse/back-projection probes;
3. pass the memory estimate for model, J-Lens, KV cache, and intervention;
4. run tiny-model baseline/capture/no-op tests;
5. run one real-model example;
6. pass five-example same-backend baseline/no-op equality;
7. show stable memory across repeated examples.

Run the primary baseline and intervention on the same qualified backend.
Results from M1/MPS, CUDA, and Radeon 8060S/ROCm are useful replication runs,
but should not be mixed into one paired comparison.

## Phase 1 — Five-example baseline smoke

1. Use `configs/smoke.yaml`, `subset_size: 5`, and capture disabled.
2. Inspect rendered prompts and generated text.
3. Confirm completion, evaluation, manifest, and report schemas.
4. Manually compare extracted answers with the generated text.
5. Record peak memory and approximate runtime.

Exit gate: all five examples complete and reevaluate without loading the model.

## Phase 2 — Capture smoke

Run the same five examples with:

1. `prompt_last` at two explicit layers;
2. `generated_last` at two explicit layers;
3. `all_generated` at two explicit layers;
4. `word_end` at two explicit layers.

Validate:

- selected rows have correct example/layer/position metadata;
- generated token IDs align with completion token IDs;
- `word_end` rows map deterministically to decoded word spans;
- hooks are removed after each run;
- GPU memory does not grow by example;
- observation-only output remains unchanged.

Exit gate: selector and token-alignment integrity checks pass.

## Phase 3 — Full baseline

1. Freeze config, environment, and dataset-selection manifest.
2. Run all examples with greedy decoding.
3. Evaluate saved completions.
4. Generate baseline report.
5. Keep the full baseline run directory immutable.

Report:

- exact-answer accuracy;
- extraction rate;
- generated token distribution;
- failed extraction examples;
- model/J-Lens/capture status;
- hardware and revision metadata.

The baseline should not be compared numerically with published scores unless
prompting and model settings are equivalent.

## Phase 4 — Full observation run

Recommended first capture:

- layers 10–26 or the resolved late band;
- `all_generated` or `word_end`;
- hidden norm, J-Space norm, and top-20 J-Space token readouts;
- no full vectors.

Compare completion hashes and accuracy against Phase 3. If they differ beyond
known same-hardware nondeterminism, investigate capture implementation before
continuing.

## Phase 5 — No-op control

Install intervention-path hooks using the `no_op` condition.

Correctness gate:

```text
for every selected example:
    baseline.generated_token_ids == no_op.generated_token_ids
    no_op.delta_hidden_norm == 0
    no_op.delta_jspace_norm == 0
```

Any mismatch blocks intervention experiments.

## Phase 6 — Strength calibration

Before a full intervention run, use a fixed calibration subset that is not
changed after inspecting results.

Suggested `mean_replace` strength sweep:

```text
0.00, 0.01, 0.02, 0.05, 0.10
```

For each strength report:

- accuracy and extraction rate;
- average relative hidden-state perturbation by layer;
- output fluency indicators such as finish reason and output length;
- number of broken and fixed examples versus baseline.

Select strengths based on preregistered perturbation/fluency criteria, not only
the largest accuracy change. Full replacement is already known from the
HumanEval reference to be model-breaking and should not be the default.

## Phase 7 — Full intervention

1. Use the frozen baseline selection and generation settings.
2. Run one condition per immutable output directory.
3. Evaluate independently.
4. Run paired comparison against the full same-hardware baseline.
5. Preserve all perturbation summaries and incompatibility warnings.

## Statistical analysis

GSM8K outcomes are paired binary observations. Report:

- accuracy in each condition;
- paired absolute accuracy difference;
- broken and fixed counts;
- exact McNemar test;
- paired bootstrap confidence interval;
- extraction failures separately.

Do not treat 1,319 examples as independent across repeated sweeps when selecting
the best-looking intervention. Strength/layer sweeps are exploratory unless a
held-out test protocol is defined.

## Interpretation boundaries

A validated intervention-induced accuracy change supports the statement that
the modified directions at the selected layers/tokens causally affect task
performance under that intervention.

It does not by itself show:

- that J-Space uniquely contains mathematical reasoning;
- that top-k coordinates are math-specific;
- that observed decoded tokens are faithful explanations;
- that the effect generalizes to other models or prompt formats.

Matched controls such as random directions/coordinates, alternative layers,
and magnitude-matched hidden-space perturbations are needed for stronger
localization claims.

## Recommended run matrix

| run | condition | capture | purpose |
|---|---|---|---|
| `smoke-baseline` | baseline | off | plumbing |
| `smoke-observe` | baseline | selected | token/layer alignment |
| `smoke-no-op` | no_op | selected | exactness gate |
| `full-baseline` | baseline | off | primary reference accuracy |
| `full-observe` | baseline | configured | normal J-Space detail |
| `calibration-a*` | intervention | configured | strength calibration |
| `full-intervention` | intervention | configured | paired primary comparison |
| `random-control` | intervention | configured | matched causal control |
