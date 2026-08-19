# qwen-humaneval-jlens — Phase 1: HumanEval baseline + J-Space observation

Phase 1 of the J-Space/J-Lens study: establish a clean HumanEval baseline for
Qwen 3.5 9B and verify that J-Space activations can be observed during
inference. The GSM8K arm of the study is run separately by a teammate.

**What this phase does**

- Loads `Qwen/Qwen3.5-9B-Base` and the fitted Jacobian lens
  (`neuronpedia/jacobian-lens@qwen-n1000`, file
  `qwen3.5-9b-pt/jlens/Salesforce-wikitext/Qwen3.5-9B-Base_jacobian_lens.pt`).
- Runs HumanEval with deterministic greedy decoding (fixed seed, fixed
  `max_new_tokens`, fixed prompt format = the raw HumanEval prompt).
- Saves prompts, raw completions, extracted code, decoding metadata.
- Computes standard HumanEval pass@1 (subprocess-executed tests).
- Captures read-only J-Space observations during generation: per hooked layer
  and token position — hidden norm, J-Space norm (`|J_l·h|`), and top-k J-Space
  token readouts (`topk(unembed(J_l·h))`, the lens's native "what is this
  activation disposed to say" view) with the associated generated token.

**What this phase does NOT do**

- Phase 1 applies no J-Space intervention: `capture_jspace.py` hooks never
  modify activations. (Phase 2 intervention now lives in
  `src/jspace_intervention.py` / `src/run_intervention.py` — see below.)

**Why the base model.** The only fitted 9B J-Lens targets
`Qwen/Qwen3.5-9B-Base`. Running the base model makes the J-Space readouts
exact rather than approximate, and completion-style HumanEval is the standard
way to evaluate base models. To use the instruct model anyway, set
`model.name: Qwen/Qwen3.5-9B` — readouts then carry a lens-mismatch caveat.

## Install

```bash
# from the parent repo root — jlens is the reference J-Lens implementation
pip install -e .

cd qwen-humaneval-jlens
pip install -r requirements.txt
```

Model, lens, and dataset download from the Hugging Face Hub on first use
(~19 GB for the model).

## Smoke test (5–10 samples)

```bash
python -m src.run_baseline --config config.yaml --limit 5
python -m src.evaluate_humaneval --config config.yaml
python -m src.analyze_baseline --config config.yaml
```

## Full HumanEval baseline (164 tasks)

```bash
# set benchmark.full_run: true in config.yaml, then:
python -m src.run_baseline --config config.yaml
python -m src.evaluate_humaneval --config config.yaml
python -m src.analyze_baseline --config config.yaml
```

The runner is resumable: completed task_ids in `completions.jsonl` are skipped
on restart.

## Outputs

- `outputs/completions/completions.jsonl` — per task: task_id, model,
  timestamp, decoding params, jlens status, prompt, raw completion, extracted
  code.
- `outputs/activations/<task_id>.jsonl.gz` — per (layer, position): token,
  hidden norm, J-Space norm, top-k J-Space token readouts.
- `outputs/evaluation/results.jsonl` + `summary.json` — per-task pass/fail and
  aggregate pass@1.
- `outputs/reports/baseline_report.md` — the Phase-1 report: pass@1,
  passed/failed examples, J-Space capture summary, J-Lens dependency status.

## Inspecting J-Space activations

```bash
# peek at one task's capture (layer, position, token, top J-Space readouts)
python - <<'EOF'
import gzip, json
with gzip.open("outputs/activations/HumanEval_0.jsonl.gz", "rt") as fh:
    for line in list(fh)[:3] + list(fh)[-3:]:
        r = json.loads(line)
        print(r["layer"], r["position"], repr(r["token"]),
              [t for t, _ in r["top_jspace_tokens"][:5]])
EOF
```

## Phase 2 — J-Space intervention (implemented: `mean_replace`)

Rerun HumanEval with an inference-time J-Space intervention and compare
against the baseline. Method: per token position, the `top_k` (default 50)
largest-|z| J-Space coordinates are replaced with the task-local running mean
J-Space activation (prefill: within-forward mean; decode: running mean over
previous positions). Only the J-Space delta is transported back via
`pinv(J_l)`, so `method: "none"` is an exact no-op control.

```bash
# smoke test (5 tasks) — first with method: "none" (must match baseline
# completions bit-for-bit on the same hardware), then "mean_replace"
python -m src.run_intervention --config config.yaml --limit 5

# evaluate + compare
python -m src.evaluate_humaneval --config config.yaml \
  --completions outputs/intervention/completions.jsonl \
  --out-dir outputs/intervention/evaluation
python -m src.compare_results --config config.yaml   # report in outputs/comparison/
```

Full run: set `benchmark.full_run: true` and drop `--limit`. Outputs:
`outputs/intervention/{completions.jsonl, run_metadata.json, hook_log.jsonl,
hook_summary.json}` and `outputs/comparison/{comparison.csv, report.md}`.
TODOs for `zero_topk`, `subtract_mean`, `random_ablation`, and
calibration-set means are marked in `src/jspace_intervention.py`.

## Results

| arm | layers | pass@1 | relative drop | broken / fixed |
|---|---|---|---|---|
| baseline (replicated ×2) | — | **0.6037** (99/164) | — | — |
| mean_replace α=0.05 k=50 | 0–9 (early) | **0.0061** (1/164) | −99.0% | 98 / 0 |
| mean_replace α=0.05 k=50 | 12–20 (mid core) | **0.3049** (50/164) | −49.5% | 51 / 2 |
| mean_replace α=0.05 k=50 | 10–26 (full band) | **0.1098** (18/164) | −81.8% | 81 / 0 |
| mean_replace α=0.05 k=50 | 27–30 (late edge) | **0.6098** (100/164) | +1.0% (noise) | 9 / 10 |

The layer map: early-layer ablation (0–9) is catastrophic but uninformative
about J-Space specificity — early representations are load-bearing for
everything, and the damage propagates through the whole network. The mid-to-late
band (10–26, core 12–20) shows the interpretable effect: fluent code, broken
logic. The final fitted layers (27–30) show no directional effect (9 broken vs
10 fixed = balanced random flips) — consistent with J-Space there being the
readout itself rather than content used by downstream computation.

Per-arm details: `outputs/intervention_layers_<first>_<last>/comparison/`.

### Phase 2 arm details

Intervention: top-50 J-Space coordinates per token moved toward the task-local
running mean with blend α=0.05, layers 10–26, all token positions. Mean
hidden-space perturbation ‖Δh‖/‖h‖ ≈ 12% per hooked layer.

| condition | pass@1 |
|---|---|
| baseline | **0.6037** (99/164) |
| intervention (mean_replace, α=0.05, k=50) | **0.1098** (18/164) |
| absolute / relative drop | **−0.4939 / −81.8%** |
| passed both / failed both / broken / fixed | 18 / 65 / 81 / 0 |

Controls: `none` reproduced the baseline completions bit-for-bit (5/5 smoke
tasks). Full-strength replacement (α=1) is model-breaking (gibberish); the
α-sweep (0.02 → 0.25) shows a dose-response: 4/5 → 1/5 → 0/5 → 0/5 on the
5-task smoke. 81 task breaks vs 0 fixes vastly exceeds the ~2-task
cross-hardware noise floor.

Interpretation: this suggests JSPace may contribute to coding-relevant
reasoning in Qwen 3.5 9B under this intervention — while noting the
perturbation is broad (17 layers × 50 coordinates), so the drop shows the
J-Space directions carry causally important content, not necessarily that
coding-specific content is uniquely localized there. Controls like
`random_ablation` (TODO in `src/jspace_intervention.py`) would sharpen that
claim.

Full details: `outputs/intervention_layers_10_26/comparison/report.md`,
per-task table in `outputs/intervention_layers_10_26/comparison/comparison.csv`.

### Phase 1: baseline (both machines)

Full HumanEval run (164 tasks), local Apple M1 Max / MPS, greedy decoding,
fitted J-Lens active for capture at layers 10–26:

| metric | value |
|---|---|
| model | `Qwen/Qwen3.5-9B-Base` |
| tasks | 164 (full benchmark) |
| baseline pass@1 | **0.6037** (99/164) |
| failures | 64 assertion/error, 1 timeout |
| J-Lens status | fitted (valid J-Space readouts) |
| activation captures | 164/164 tasks, layers 10–26 |

_See `outputs/reports/baseline_report.md` for the machine-generated report
with per-layer activation norms, example readouts, and the failed-task list._

**Cross-hardware replication (RTX 5090 Laptop / CUDA):** the identical pipeline
on the Olares GPU host reproduced pass@1 = **0.6037** (99/164). 141/164 raw
completions are bit-identical; two tasks flipped (HumanEval/102 local-fail →
remote-pass, HumanEval/146 local-pass → remote-fail) — ordinary bf16
nondeterminism across GPU architectures. Remote artifacts:
`outputs-remote-5090/`. Phase-2 intervention effects should be read against
this ~1-task noise floor.

## Warning

`evaluate_humaneval.py` executes model-generated Python in a subprocess
(10 s timeout). Run it in an environment you trust.
