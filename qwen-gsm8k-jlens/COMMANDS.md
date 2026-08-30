# Command guide

How to run GSM8K, BIPIA, AgentDojo, and InjecAgent from
`qwen-gsm8k-jlens/`. The package name is still `gsm8k_jspace`; the
benchmark is selected with `benchmark.name` in YAML, not by a different
CLI.

All commands below assume you are in this directory:

```bash
cd qwen-gsm8k-jlens
```

## Two ways to invoke

### 1. `scripts/uv-env` (recommended on GPU machines)

Picks `.venv-mlx` / `.venv-rocm` / `.venv-cuda` / `.venv-cpu` from the
host, then runs a **fixed** subset of commands:

| `uv-env` command | what it actually runs |
|---|---|
| `./scripts/uv-env sync` | install the matching venv |
| `./scripts/uv-env diagnose [--config …]` | `python -m gsm8k_jspace diagnose` |
| `./scripts/uv-env run --config …` | `python -m gsm8k_jspace run` |
| `./scripts/uv-env jupyter` | Jupyter Lab |

Override host detection:

```bash
./scripts/uv-env --profile cpu sync
./scripts/uv-env --profile m1-max diagnose --config configs/smoke.yaml
```

`uv-env run` is **only** the `run` subcommand. Do not put
`python -m gsm8k_jspace inspect-config` after it; that would be passed
as extra arguments to `run`.

### 2. `python -m gsm8k_jspace` (full CLI)

After `./scripts/uv-env sync`, use the venv Python for every
subcommand (`inspect-config`, `diagnose`, `run`, `evaluate`,
`compare`):

```bash
# Apple Silicon example; swap in .venv-rocm / .venv-cuda / .venv-cpu
.venv-mlx/bin/python -m gsm8k_jspace --help
```

Or, with the environment already selected:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-mlx"   # or .venv-rocm / .venv-cuda
uv run --no-sync python -m gsm8k_jspace inspect-config --config configs/smoke.yaml
```

The rest of this file uses `python -m gsm8k_jspace …`. On a machine
where you already synced, you can replace `python -m gsm8k_jspace run`
with `./scripts/uv-env run`.

---

## Setup once

```bash
chmod +x scripts/uv-env scripts/detect-host.py scripts/verify-environment.py
./scripts/uv-env sync
./scripts/uv-env diagnose --config configs/smoke.yaml
```

`diagnose` prints host profile, resolved backend, dtype, device name,
warnings, and a memory estimate. It does not load the full model
weights for generation.

Pass `--config` so the estimate matches the YAML you intend to run
(model size, capture, GPUs).

---

## Shared flags (`inspect-config` and `run`)

These flags apply to both `inspect-config` and `run`. Later files win
on key collisions: `--config` is merged first, then `--host-config`,
then each `--overlay` in order, then CLI flags.

| flag | meaning |
|---|---|
| `--config PATH` | **Required.** Base YAML. |
| `--host-config PATH` | Optional host overlay (`configs/hosts/apple.yaml`, `amd.yaml`, `nvidia.yaml`). |
| `--overlay PATH` | Optional extra YAML. Repeatable. |
| `--limit N` | Cap examples after dataset selection (`benchmark.subset_size` still applies first). |
| `--run-id LABEL` | Experiment label. A UTC stamp `YYYYMMDDTHHMMSSZ` is appended unless the value already ends with that pattern. |
| `--resume` | Force resume into an existing run folder (also set by `outputs.on_existing: resume` in YAML). |
| `--condition NAME` | Override `experiment.condition` (`baseline`, `no_op`, `intervention`, …). |
| `--gpus 0,1` | Physical GPU indices. More than one GPU enables parallel workers; each loads a **full model copy**. |
| `--capture` | Turn J-Space recording on (`capture.enabled: true`). |
| `--no-capture` | Skip capture files; completions are still written. |
| `--evaluate` | **`run` only.** Score the run immediately after generation. |

`--capture` and `--no-capture` are mutually exclusive. If neither is
passed, the YAML `capture.enabled` value is used.

---

## `inspect-config`

Print the fully resolved YAML. Does **not** download data or load a
model. Use this to confirm `benchmark.name`, parser, output root, and
overlays before a long run.

```bash
python -m gsm8k_jspace inspect-config --config configs/smoke.yaml
python -m gsm8k_jspace inspect-config --config configs/bipia-smoke.yaml
python -m gsm8k_jspace inspect-config \
  --config configs/runs/small-smoke.yaml \
  --host-config configs/hosts/apple.yaml \
  --overlay configs/experiments/mean-replace.yaml
```

Check that parser and benchmark match:

| `benchmark.name` | required `evaluation.parser` |
|---|---|
| `gsm8k` | `gsm8k_numeric_v1` |
| `bipia` | `bipia_asr_v1` |
| `agentdojo` | `agentdojo_asr_v1` |
| `injecagent` | `injecagent_asr_v1` |

---

## `run`

Load the model, generate one completion per selected example, write
artifacts under `outputs.<root_dir>/<run_id>/`.

### GSM8K smoke (default)

```bash
# no J-Space capture files
python -m gsm8k_jspace run --config configs/smoke.yaml --evaluate --no-capture

# record J-Space top-k / norms
python -m gsm8k_jspace run --config configs/smoke.yaml --evaluate --capture
```

Equivalent with the wrapper:

```bash
./scripts/uv-env run --config configs/smoke.yaml --evaluate --no-capture
```

Outputs: `outputs/gsm8k/<run_id>/`.

### BIPIA smoke

```bash
python -m gsm8k_jspace run --config configs/bipia-smoke.yaml --evaluate --no-capture
python -m gsm8k_jspace run --config configs/bipia-smoke.yaml --evaluate --capture --limit 2
```

Outputs: `outputs/bipia/<run_id>/`. Local rule/heuristic ASR, not
official GPT-4 judges. The summary includes `asr_method` so the two
are not mixed.

### AgentDojo smoke

```bash
python -m gsm8k_jspace run --config configs/agentdojo-smoke.yaml --evaluate --no-capture
```

Outputs: `outputs/agentdojo/<run_id>/`. First-turn ASR after a
poisoned tool observation. Not the full AgentDojo gym / utility loop.

### InjecAgent smoke

```bash
python -m gsm8k_jspace run --config configs/injecagent-smoke.yaml --evaluate --no-capture
```

Outputs: `outputs/injecagent/<run_id>/`. First-step tool-call ASR
only (no data-stealing step 2).

### Overlay a condition instead of flags

```bash
# capture on, still GSM8K smoke
python -m gsm8k_jspace run \
  --config configs/smoke.yaml \
  --overlay configs/experiments/capture-on.yaml \
  --evaluate

# baseline + mean-replace intervention on the small GSM8K tier
python -m gsm8k_jspace run \
  --config configs/runs/small-smoke.yaml \
  --host-config configs/hosts/apple.yaml \
  --overlay configs/experiments/mean-replace.yaml \
  --evaluate

# BIPIA with the same mean-replace overlay
python -m gsm8k_jspace run \
  --config configs/bipia-smoke.yaml \
  --overlay configs/experiments/mean-replace.yaml \
  --evaluate
```

Condition overlays under `configs/experiments/`:

| overlay | effect |
|---|---|
| `baseline.yaml` | no intervention |
| `no-op.yaml` | exact no-op hook control |
| `mean-replace.yaml` | J-Space `mean_replace` |
| `capture-on.yaml` / `capture-off.yaml` | recording on/off |
| `capture-all-layers-inference.yaml` | denser capture |

### Ready-made 9B GSM8K YAMLs

Self-contained model + host + GSM8K. See `configs/README.md` for the
full table. Examples:

```bash
./scripts/uv-env run --config configs/radeon-qwen35-9b-smoke-basic.yaml --evaluate
./scripts/uv-env run --config configs/nvidia-qwen35-9b-500-jlens.yaml --evaluate
./scripts/uv-env run --config configs/nvidia-qwen35-9b-500-basic.yaml --gpus 0,2 --evaluate
```

### Limit, GPUs, condition

```bash
python -m gsm8k_jspace run --config configs/smoke.yaml --limit 3 --evaluate --no-capture
python -m gsm8k_jspace run --config configs/nvidia-qwen35-9b-500-basic.yaml --gpus 1 --evaluate
python -m gsm8k_jspace run --config configs/smoke.yaml --condition baseline --evaluate
```

---

## Run folders and `--run-id`

Every run gets a UTC-stamped folder so reruns do not overwrite each
other, for example:

```
m1max-mlx-qwen35-4b-smoke5_20260820T001408Z
```

`--run-id` is a **label**. If you pass `my-bipia-smoke`, the folder
becomes `my-bipia-smoke_YYYYMMDDTHHMMSSZ`. If the value already ends
with `YYYYMMDDTHHMMSSZ`, it is used as the full folder name (this is
how resume works).

```bash
python -m gsm8k_jspace run \
  --config configs/bipia-smoke.yaml \
  --run-id bipia-email-qwen35-9b \
  --evaluate --no-capture
```

---

## Resume after stop / pause

Progress is written to `progress.json` after each finished example.
Completed IDs live in `completions.jsonl`. Ready-made configs set
`outputs.on_existing: resume`.

To continue, pass the **full folder name** (label + timestamp):

```bash
python -m gsm8k_jspace run \
  --config configs/nvidia-qwen35-9b-500-basic.yaml \
  --run-id nvidia-qwen35-9b-500-basic_20260821T153012Z \
  --evaluate
```

Inspect progress:

```bash
cat outputs/gsm8k/<run_id>/progress.json
cat outputs/bipia/<run_id>/progress.json
```

GSM8K run fingerprints omit unused nested benchmark sections so an
old GSM8K resume still matches after BIPIA / AgentDojo / InjecAgent
keys were added to the schema. Changing `benchmark.name`, parser,
model, or selection still starts a new run.

---

## `evaluate`

Score an existing run directory. Uses `run_config.resolved.yaml` in
that folder (you do not pass `--config`).

```bash
python -m gsm8k_jspace evaluate --run outputs/gsm8k/<run_id>
python -m gsm8k_jspace evaluate --run outputs/bipia/<run_id>
python -m gsm8k_jspace evaluate --run outputs/agentdojo/<run_id>
python -m gsm8k_jspace evaluate --run outputs/injecagent/<run_id>
```

GSM8K writes exact-answer accuracy. The injection suites write ASR
plus `asr_method` / attack metadata.

Use this when you generated without `--evaluate`, or when you changed
the evaluator and want to rescore the same completions.

---

## `compare`

Diff two completed run directories (metrics + fingerprint / config
drift). Pair baseline and intervention only on the **same GPU class**.

```bash
python -m gsm8k_jspace compare \
  --baseline outputs/gsm8k/<baseline_run_id> \
  --candidate outputs/gsm8k/<intervention_run_id>

python -m gsm8k_jspace compare \
  --baseline outputs/bipia/<baseline_run_id> \
  --candidate outputs/bipia/<intervention_run_id>
```

---

## What a run directory contains

Typical files under `outputs/<benchmark>/<run_id>/`:

| file | contents |
|---|---|
| `run_config.resolved.yaml` | fully merged config used for the run |
| `completions.jsonl` | one generated example per line |
| `progress.json` | finished IDs / resume cursor |
| `metrics.json` / evaluation summary | accuracy or ASR |
| capture sidecars | J-Space top-k / norms when `--capture` is on |

Open `notebooks/00-run-overview.ipynb` and set `RUN_DIR` to that
folder. Notebooks read artifacts only; they do not load the 9B model.

```bash
./scripts/uv-env jupyter
```

---

## Tests (no Qwen download)

```bash
PYTHONPATH=src:.. pytest tests -q
```

Fast tests use synthetic rows and the parent repo’s `tests/tiny.py`.

---

## Worked sequences

### A. First BIPIA smoke on this machine

```bash
./scripts/uv-env sync
python -m gsm8k_jspace inspect-config --config configs/bipia-smoke.yaml
./scripts/uv-env diagnose --config configs/bipia-smoke.yaml
./scripts/uv-env run --config configs/bipia-smoke.yaml --evaluate --no-capture
python -m gsm8k_jspace evaluate --run outputs/bipia/<the_new_folder>
```

### B. GSM8K baseline vs mean-replace (same host)

```bash
python -m gsm8k_jspace run \
  --config configs/runs/small-smoke.yaml \
  --host-config configs/hosts/apple.yaml \
  --overlay configs/experiments/baseline.yaml \
  --evaluate --no-capture

python -m gsm8k_jspace run \
  --config configs/runs/small-smoke.yaml \
  --host-config configs/hosts/apple.yaml \
  --overlay configs/experiments/mean-replace.yaml \
  --evaluate --capture

python -m gsm8k_jspace compare \
  --baseline outputs/gsm8k/<baseline_folder> \
  --candidate outputs/gsm8k/<intervention_folder>
```

### C. InjecAgent, two examples only, then rescore later

```bash
python -m gsm8k_jspace run \
  --config configs/injecagent-smoke.yaml \
  --limit 2 --no-capture

python -m gsm8k_jspace evaluate --run outputs/injecagent/<run_id>
```
