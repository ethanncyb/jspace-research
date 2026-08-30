# Terminal launchers

Shell equivalents of [`notebooks/JSpace_End_to_End_Local.ipynb`](../notebooks/JSpace_End_to_End_Local.ipynb).

## Quick start

```bash
# 1. Edit the launcher config (experiment YAML, GPU, output path)
vim scripts/config.yaml

# 2. Export API keys (or point to token files in config.yaml)
export OPENROUTER_API_KEY='your-key'
export HF_TOKEN='your-hf-token'

# 3. Run
./scripts/setup.sh          # once per machine (repos + Python deps)
./scripts/run_all.sh        # phases 1–4
```

Default settings use Qwen 3.5 9B smoke:

- experiment config: `configs/phase1_qwen35_9b_smoke.yaml`
- output: `artifacts/jspace-qwen35_9b-smoke-gpu0/`

## Launcher config

Edit [`scripts/config.yaml`](config.yaml) to change:

| Section | Purpose |
| --- | --- |
| `experiment.config` | Scientific phase YAML under `configs/` |
| `hardware.physical_gpu_index` | Physical GPU remapped to logical `cuda:0` |
| `output.run_root` | Artifact root (`null` = auto-derive from config name + GPU) |
| `paths` | BIPIA checkout, benchmark roots, optional train data |
| `credentials` | Token file paths (optional; env vars work too) |
| `runtime` | Virtualenv settings |
| `pipeline` | `skip_setup`, `skip_gpu_check`, default `stage` |

For machine-specific overrides without editing the committed config, copy snippets into `scripts/config.local.yaml` (gitignored). That file is merged on top of `config.yaml`.

Reference copy with comments: [`config.example.yaml`](config.example.yaml).

### Example: switch model or GPU

```yaml
experiment:
  config: configs/phase1_smoke.yaml   # Gemma smoke

hardware:
  physical_gpu_index: 2               # use GPU 2 on a 4x L4 node

output:
  run_root: artifacts/my-gemma-smoke-gpu2
```

## Scripts

| Script | Purpose |
| --- | --- |
| `download_repos.sh` | Clone/init BIPIA, AgentDojo, InjecAgent at pinned revisions |
| `setup.sh` | `download_repos.sh` + install Python package + verify CLI |
| `run_all.sh` | Phases 1–4 end to end (setup optional via `pipeline.skip_setup`) |
| `phase1.sh` | Phase 1 (`STAGE=prepare\|capture\|analyze\|all`) |
| `phase2.sh` | Phase 2 (`STAGE=generate\|analyze`) |
| `phase3.sh` | Phase 3 detectors |
| `phase4.sh` | Phase 4 transfer (`STAGE=generate\|analyze\|all`) |

All scripts accept `--config PATH` (default: `scripts/config.yaml`).

```bash
./scripts/download_repos.sh   # repos only
./scripts/setup.sh            # repos + Python install
./scripts/phase1.sh
STAGE=analyze ./scripts/phase2.sh
JSPACE_PHYSICAL_GPU_INDEX=2 ./scripts/phase1.sh
./scripts/run_all.sh --config scripts/config.yaml
```

## Environment variables

Environment variables override YAML values.

| Variable | Default | Purpose |
| --- | --- | --- |
| `JSPACE_CONFIG_PATH` | from `experiment.config` | Scientific phase YAML |
| `JSPACE_PHYSICAL_GPU_INDEX` | `0` | Physical GPU remapped to logical `cuda:0` |
| `JSPACE_RUN_ROOT` | derived | Artifact root |
| `JSPACE_BENCHMARKS_ROOT` | `../jspace-benchmarks` | AgentDojo / InjecAgent checkouts |
| `JSPACE_USE_PROJECT_VENV` | `1` | Use `.venv`; set `0` on restricted hubs |
| `WEBQA_TRAIN_PATH` | unset | Required for full Phase 1 |
| `SUMMARIZATION_TRAIN_PATH` | unset | Required for full Phase 1 |
| `STAGE` | phase-specific | Override stage for per-phase scripts |
| `SKIP_SETUP` | `0` | Set `1` in `run_all.sh` to skip checkout/setup |
| `SKIP_GPU_CHECK` | `0` | Set `1` to skip the CUDA preflight in `run_all.sh` |

Pipeline commands use `python -m jspace_research.phaseN.cli`, so they do not require `jspace-phase*` entry points on `PATH`.

## Multi-GPU nodes

The pipeline runs on a single GPU (`cuda:0`). Set `hardware.physical_gpu_index` to pick which physical card is remapped to `cuda:0`. To run two experiments in parallel on a multi-GPU machine, use separate terminal sessions with different `physical_gpu_index` and `output.run_root` values.

## Legacy launcher configs

Older two-file launchers (`scripts/config/run.yaml` + `scripts/config/env.yaml`) are still supported via `--run-config` and `--env-config`. Prefer the unified `scripts/config.yaml`.
