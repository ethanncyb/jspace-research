# GSM8K × J-Space benchmark

Python benchmark for Qwen with J-Space observation and intervention, now
pluggable across **GSM8K**, **BIPIA**, **AgentDojo**, and **InjecAgent**.

Shared across every benchmark:

- a normal / observation-only baseline
- an exact no-op hook control
- a configurable J-Space intervention (`mean_replace`)
- saved J-Space details (layers, tokens, words, last token)
- Jupyter notebooks over saved artifacts

Set `benchmark.name` in YAML. GSM8K still measures **exact-answer accuracy**.
The injection suites measure **attack success rate (ASR)**. Teacher-forced
next-token controllability remains `benchmark.test_type: gold_next_token`
(GSM8K only).

## Benchmarks

| `benchmark.name` | metric | smoke config | data |
|---|---|---|---|
| `gsm8k` | exact-answer accuracy | `configs/smoke.yaml` | Hugging Face `openai/gsm8k` |
| `bipia` | ASR (rule/heuristic) | `configs/bipia-smoke.yaml` | `data/bipia/` (email/table/code) |
| `agentdojo` | first-turn ASR | `configs/agentdojo-smoke.yaml` | `data/agentdojo/workspace_static.json` |
| `injecagent` | first-step tool-call ASR | `configs/injecagent-smoke.yaml` | `data/injecagent/` |

AgentDojo and InjecAgent run as **single-generation** slices that match this
project's capture/intervention loop: the model sees a poisoned tool
observation and we score the next action. That is not the full AgentDojo
environment loop or InjecAgent data-stealing step 2.

BIPIA scoring is local (`bipia_asr_v1`). Official BIPIA uses GPT-4 judges for
most text attacks; reports include `asr_method` so the two are not mixed.

## Layout

- `COMMANDS.md` — CLI flags, resume, evaluate/compare, and per-benchmark examples
- `configs/` — default, smoke, host, run-tier, and condition YAML
- `configs/README.md` — ready-made Radeon / NVIDIA Qwen3.5-9B GSM8K run YAMLs
- `data/` — BIPIA / AgentDojo / InjecAgent files used by non-GSM8K configs
- `src/gsm8k_jspace/` — typed config, runner, capture, intervention, evaluation
- `src/gsm8k_jspace/benchmarks/` — plugin registry (`gsm8k`, `bipia`, `agentdojo`, `injecagent`)
- `notebooks/` — parameterized analysis notebooks
- `scripts/uv-env` — pick `.venv-mlx` / `.venv-rocm` / `.venv-cuda` / `.venv-cpu`
- `plans/` — design documents

## Setup (uv)

From this directory:

```bash
chmod +x scripts/uv-env scripts/detect-host.py scripts/verify-environment.py
./scripts/uv-env sync
./scripts/uv-env diagnose --config configs/smoke.yaml
```

`scripts/uv-env` detects the host GPU and sets `UV_PROJECT_ENVIRONMENT`:

| host | environment | backend |
|---|---|---|
| M1 Max MacBook Pro | `.venv-mlx` | MLX (`mlx` + `mlx-lm`) |
| Framework desktop, Radeon 8060S | `.venv-rocm` | ROCm |
| A100 / H100 | `.venv-cuda` | CUDA |
| CPU / CI | `.venv-cpu` | CPU |

Apple Silicon uses **MLX** by default. `./scripts/uv-env sync` installs the
`apple` extra. PyTorch MPS remains available with `./scripts/uv-env --profile mps`.

Override detection with `./scripts/uv-env --profile cpu sync`.

The parent `jlens` package is installed as an editable path dependency.

## Commands

See **[COMMANDS.md](COMMANDS.md)** for flags, resume, evaluate/compare,
and examples for every benchmark. Quick start:

```bash
python -m gsm8k_jspace inspect-config --config configs/smoke.yaml
python -m gsm8k_jspace run --config configs/smoke.yaml --evaluate --no-capture
python -m gsm8k_jspace run --config configs/bipia-smoke.yaml --evaluate --no-capture
python -m gsm8k_jspace run --config configs/agentdojo-smoke.yaml --evaluate --no-capture
python -m gsm8k_jspace run --config configs/injecagent-smoke.yaml --evaluate --no-capture
```

On GPU hosts, `./scripts/uv-env run --config …` is the same as
`python -m gsm8k_jspace run`. Output folders are UTC-stamped so reruns
do not overwrite each other. `--run-id` is a label; pass the full
folder name to resume.

## Run tiers

1. **Small (M1 Max):** 5 examples, `configs/runs/small-smoke.yaml`
2. **Medium (Radeon 8060S):** 100 examples, `configs/runs/medium-validation.yaml`
3. **Large (A100/H100):** full GSM8K test, `configs/runs/large-full.yaml`

Pair baseline and intervention only on the **same GPU class**.

## Tests

```bash
PYTHONPATH=src:.. pytest tests -q
```

Fast tests use synthetic GSM8K rows and `tests/tiny.py` from the parent repo.
They do not download Qwen.

## Jupyter

```bash
./scripts/uv-env jupyter
```

Open `notebooks/00-run-overview.ipynb` and set `RUN_DIR` to a completed run.
Notebooks read saved artifacts only; they do not load the 9B model.
