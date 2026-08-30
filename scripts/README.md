# Terminal launchers

Shell equivalents of [`notebooks/JSpace_End_to_End_Local.ipynb`](../notebooks/JSpace_End_to_End_Local.ipynb).

## Quick start

### Option A: YAML configs (recommended)

```bash
cp scripts/config/run.example.yaml scripts/config/run.yaml
cp scripts/config/env.example.yaml scripts/config/env.yaml
# edit run.yaml (experiment config) and env.yaml (paths + token files)

./scripts/setup.sh
./scripts/run_e2e.sh
```

### Option B: environment variables only

```bash
export OPENROUTER_API_KEY='your-key'
export HF_TOKEN='your-hf-token'

./scripts/setup.sh
./scripts/run_e2e.sh
```

Default run settings match the local notebook:

- experiment config: `configs/phase1_qwen35_9b_smoke.yaml`
- output: `artifacts/jspace-qwen35_9b-smoke-gpu0/`

## YAML configs

Two launcher files live under `scripts/config/`:

| File | Purpose |
| --- | --- |
| `run.yaml` | Which scientific phase YAML to run, GPU index, run root, pipeline flags |
| `env.yaml` | Machine paths, venv settings, and credential **file** locations |

Committed templates:

- [`run.example.yaml`](config/run.example.yaml)
- [`env.example.yaml`](config/env.example.yaml)

Copy them to `run.yaml` and `env.yaml` (gitignored). Environment variables always override YAML.

Example `run.yaml`:

```yaml
experiment:
  config: configs/phase1_qwen35_9b_smoke.yaml
  physical_gpu_index: 0
```

Example `env.yaml`:

```yaml
runtime:
  use_project_venv: false   # TLJH / JupyterHub

paths:
  benchmarks_root: /opt/dlami/nvme/jspace-benchmarks

credentials:
  hf_token_file: ~/.config/jspace/hf_token
  openrouter_api_key_file: ~/.config/jspace/openrouter_key
```

Pass explicit paths on the command line:

```bash
./scripts/run_e2e.sh \
  --run-config scripts/config/run.yaml \
  --env-config scripts/config/env.local.yaml
```

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `JSPACE_RUN_CONFIG` | `scripts/config/run.yaml` if present | Launcher run YAML |
| `JSPACE_ENV_CONFIG` | `scripts/config/env.yaml` if present | Launcher env YAML |
| `JSPACE_MODEL_KEY` | `qwen35_9b` | Config/model prefix when `experiment.config` omitted |
| `JSPACE_RUN_MODE` | `smoke` | `smoke` or `full` when `experiment.config` omitted |
| `JSPACE_CONFIG_PATH` | derived | Direct path to phase YAML |
| `JSPACE_PHYSICAL_GPU_INDEX` | `0` | Physical GPU remapped to logical `cuda:0` |
| `JSPACE_RUN_ROOT` | derived | Override artifact root |
| `JSPACE_BENCHMARKS_ROOT` | `../jspace-benchmarks` | AgentDojo / InjecAgent checkouts |
| `JSPACE_USE_PROJECT_VENV` | `1` | Use `.venv`; set `0` on restricted hubs |
| `JSPACE_PYTHON` | auto | Force a specific Python executable |
| `WEBQA_TRAIN_PATH` | unset | Required for full Phase 1 |
| `SUMMARIZATION_TRAIN_PATH` | unset | Required for full Phase 1 |
| `STAGE` | phase-specific | Override stage for per-phase scripts |
| `SKIP_SETUP` | `0` | Set `1` in `run_e2e.sh` to skip checkout/setup |
| `SKIP_GPU_CHECK` | `0` | Set `1` to skip the CUDA preflight |

Examples:

```bash
JSPACE_PHYSICAL_GPU_INDEX=2 ./scripts/run_phase1.sh

./scripts/run_e2e.sh --run-config scripts/config/run.yaml --env-config scripts/config/env.yaml

STAGE=analyze ./scripts/run_phase2.sh
```

## Scripts

| Script | Notebook section | Purpose |
| --- | --- | --- |
| `setup.sh` | 1 | Install package, init BIPIA submodule, clone benchmark pins |
| `run_e2e.sh` | 4–13 | Full Phases 1–4 |
| `run_phase1.sh` | 6 | Phase 1 only (`STAGE=prepare\|capture\|analyze\|all`) |
| `run_phase2.sh` | 8–9 | Phase 2 (`STAGE=generate\|analyze`) |
| `run_phase3.sh` | 11 | Phase 3 detectors |
| `run_phase4.sh` | 12 | Phase 4 transfer (`STAGE=generate\|analyze\|all`) |

Pipeline commands use `python -m jspace_research.phaseN.cli`, so they do not require `jspace-phase*` entry points on `PATH`.
