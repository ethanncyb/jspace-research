# 10 — Decisions, risks, and open questions

This document tracks choices that must be frozen before implementation or a
full experiment. They must not be resolved implicitly by whichever host runs
first.

## Decisions already made

| area | decision |
|---|---|
| implementation | Python package with typed configuration |
| environments | uv with isolated MPS, ROCm, CUDA, and CPU environments |
| primary metric | full-answer GSM8K exact numeric accuracy |
| secondary metric | optional teacher-forced next-token controllability |
| conditions | baseline, observation-only baseline, exact no-op, intervention |
| execution tiers | M1 Max small, Radeon 8060S medium, A100/H100 full |
| analysis | reusable Python analysis API plus parameterized Jupyter notebooks |
| outputs | immutable run artifacts plus executed notebooks/HTML/figures/tables |
| comparison | paired same-hardware analysis; cross-hardware runs are replications |
| causal controls | no-op and matched random controls are mandatory |

## Decision gate 1 — Model and exact J-Lens

### Recommended default

Use `Qwen/Qwen3.5-9B-Base` with the fitted
`Qwen3.5-9B-Base_jacobian_lens.pt`.

Reason: the existing lens is fitted for the base model. Using an instruct model
with this checkpoint creates a model/lens mismatch that weakens J-Space claims.

### Alternative

Use an instruct model only if:

- a matching fitted J-Lens is available; or
- the run is explicitly labeled exploratory/lens-mismatched and excluded from
  the primary result.

Freeze:

- model and tokenizer repository/revision;
- J-Lens repository/revision/file checksum;
- trust-remote-code policy;
- dtype per backend.

## Decision gate 2 — Canonical GSM8K prompt

The current draft names `zero_shot_cot_v1`, but that is not yet a justified
canonical protocol for a base completion model.

Candidate protocols:

1. fixed zero-shot completion prompt;
2. fixed few-shot GSM8K exemplars with chain-of-thought;
3. a matching instruct/chat-template protocol, only with a compatible lens.

### Required pilot

On the Radeon medium tier, compare zero-shot and one fixed few-shot protocol on
a prompt-selection subset that is not reused for the final test claim.

Select one protocol using preregistered criteria:

- answer extraction rate;
- nonempty/valid completion rate;
- context length and runtime;
- baseline accuracy;
- consistency across examples.

After selection:

- assign a versioned template name;
- save exact exemplar IDs/text and order;
- freeze whitespace and answer instruction;
- use the same rendered prompts for baseline/no-op/intervention;
- do not choose between prompt variants using final full-run results.

## Decision gate 3 — Context overflow

Policy must be explicit because few-shot prompting plus long rationales can
exceed configured context.

Recommended policy:

1. compute prompt tokens plus `max_new_tokens` before generation;
2. never left/right truncate a GSM8K question silently;
3. reduce few-shot exemplars according to a fixed versioned rule;
4. if the question still cannot fit, record `context_overflow` and exclude it
   from generation while keeping it in denominator/accounting reports;
5. use the identical effective prompt in every compared condition.

The completion record saves context limit, original/effective prompt length,
and any exemplar reduction.

## Decision gate 4 — Capture timing and token alignment

KV-cached generation creates an alignment issue: a hook observes the token fed
into a forward pass, while that pass emits logits for the next token. The final
emitted token may not receive another model forward before generation stops.

Required definitions:

- `state_token_position`: token represented by the captured hidden state;
- `predicted_token_position`: token selected from that state's output logits;
- `generated_position`: generated token index where applicable;
- `capture_event`: `prefill`, `decode`, or explicit final replay.

For `generated_last`, choose and version one method:

- final replay: run a non-generating forward over the completed sequence to
  observe the last emitted token; recommended for exact semantics;
- last decode state: observe the state that predicted the final token and label
  it as such, not as the final token's activation.

The same distinction applies to every-word/`word_end` capture.

## Decision gate 5 — Intervention token scope

The HumanEval reference calls its policy `all_generated_tokens`, but its hooks
also modify every prompt-prefill position.

GSM8K configuration must distinguish:

- `prompt_positions`;
- `decode_input_positions`;
- `state_predicting_generated_tokens`;
- `final_replay_positions`.

Recommended initial experiments:

1. generated/decode positions only;
2. prompt plus generated positions as a separately named condition.

Never use one label for both scopes.

## Decision gate 6 — Primary intervention and controls

`mean_replace` is the first reusable method, but an effect from top-absolute
coordinates is not specific without controls.

The frozen full-run matrix includes:

- baseline;
- exact no-op;
- `mean_replace`;
- random-coordinate control matched on layer, token count, coordinate count,
  and intervention norm;
- where feasible, random-direction hidden-space control matched on hidden-state
  perturbation norm.

Strength is selected on the medium calibration tier using output-validity and
perturbation criteria, not maximum observed accuracy damage.

## Decision gate 7 — Dataset selection and revisions

Freeze:

- `openai/gsm8k`, `main`, `test`;
- dataset revision/commit where the loader exposes it;
- stable source indices and question hashes;
- M1 5–10-example manifest;
- Radeon 100-example calibration/validation manifest;
- full 1,319-example manifest.

The prompt-selection/calibration subset and final confirmatory interpretation
must be separated. If the same full dataset is used after tuning strengths,
label the result exploratory rather than confirmatory.

## Decision gate 8 — Capture storage budget

Before full runs, estimate:

```text
examples × generated positions × layers × bytes per capture row
```

Freeze:

- captured layers;
- token selector;
- top-k vocabulary size;
- whether pre/post intervention states are both saved;
- full-vector examples/layers;
- JSONL-gzip versus Parquet;
- maximum expected run size.

Recommended final default:

- norms and top-k readouts for selected layers/tokens;
- no full vectors for all 1,319 examples;
- full vectors only for a fixed diagnostic subset;
- per-example sharding and checksums.

## Decision gate 9 — A100 versus H100

Choose one GPU class for the primary paired run based on availability. Do not
run baseline on A100 and intervention on H100.

If both are available:

- designate one as primary;
- repeat a fixed shared subset on the other as replication;
- record CUDA/driver/torch/dtype differences;
- report completion agreement before comparing aggregate accuracy.

## Decision gate 10 — Notebook publication format

Freeze which notebooks are generated for every completed run and which are
comparison-only.

Recommended:

- every run: overview and accuracy;
- capture run: J-Space capture notebook;
- paired run: intervention comparison and example explorer;
- experiment suite: cross-hardware replication;
- always export executed `.ipynb` and standalone HTML;
- save PNG/SVG figures and backing CSV tables.

## Risk register

| risk | consequence | mitigation/gate |
|---|---|---|
| base model prompt performs poorly | benchmark measures prompting failure | prompt pilot and frozen version |
| instruct model uses mismatched lens | invalid J-Space interpretation | exact model/lens compatibility gate |
| final token capture is mislabeled | wrong token-level conclusions | explicit state/prediction positions or final replay |
| prompt tokens modified unintentionally | intervention scope differs from claim | separate selectors and condition names |
| strength selected on final data | optimistic effect estimate | medium-tier calibration and frozen full run |
| random control omitted | effect not specific to J-Space feature choice | mandatory matched control |
| context silently truncated | conditions/tasks change invisibly | explicit overflow records and no silent truncation |
| cross-hardware conditions paired | numeric/device effects confound intervention | same-hardware primary pairing |
| capture exceeds storage/memory | incomplete or unstable long run | preflight size estimate and bounded vector capture |
| notebooks contain unique metric code | results become manually irreproducible | Python analysis API and fixture execution |

## Freeze checklist before full A100/H100 run

- [ ] Exact model, tokenizer, and J-Lens revisions/checksums.
- [ ] Canonical prompt template and exemplar set.
- [ ] Dataset manifest and calibration/final interpretation.
- [ ] Context overflow policy.
- [ ] Generation settings and stopping policy.
- [ ] Capture layer/token/timing semantics.
- [ ] Final-token replay policy.
- [ ] Intervention scope, method, strength, and layers.
- [ ] No-op and random matched controls.
- [ ] Storage estimate and artifact format.
- [ ] Primary GPU class and locked uv environment.
- [ ] Answer parser version.
- [ ] Notebook/export set.
- [ ] Paired statistical analysis version.

No full-run result should be interpreted until this checklist is saved with the
experiment preregistration/configuration.
