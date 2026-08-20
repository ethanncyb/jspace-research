# GSM8K × J-Space benchmark

Python benchmark for Qwen on GSM8K with:

- a normal / observation-only baseline
- an exact no-op hook control
- a configurable J-Space intervention (`mean_replace`)
- saved J-Space details (layers, tokens, words, last token)
- Jupyter notebooks over saved artifacts

This is the GSM8K counterpart of the HumanEval J-Space study. It measures
**exact-answer accuracy**, not teacher-forced next-token controllability.
The latter remains available as `benchmark.test_type: gold_next_token`.

## Layout

- `configs/` — default, smoke, host, run-tier, and condition YAML
- `configs/README.md` — commands for ready-made Radeon / NVIDIA Qwen3.5-9B runs
- `src/gsm8k_jspace/` — typed config, runner, capture, intervention, evaluation
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

```bash
# print the fully resolved config (no model load)
python -m gsm8k_jspace inspect-config --config configs/smoke.yaml

# small run without J-Space capture files
python -m gsm8k_jspace run --config configs/smoke.yaml --evaluate --no-capture

# same run, but record J-Space top-k / norms
python -m gsm8k_jspace run --config configs/smoke.yaml --evaluate --capture

# overlay instead of flags
python -m gsm8k_jspace run --config configs/smoke.yaml --overlay configs/experiments/capture-on.yaml

# merge a condition overlay
python -m gsm8k_jspace run \
  --config configs/runs/small-smoke.yaml \
  --host-config configs/hosts/apple.yaml \
  --overlay configs/experiments/mean-replace.yaml

python -m gsm8k_jspace evaluate --run outputs/gsm8k/<run_id>
python -m gsm8k_jspace compare --baseline <baseline_dir> --candidate <run_dir>
```

Output folders are UTC-stamped so reruns do not overwrite each other, for
example `m1max-mlx-qwen35-4b-smoke5_20260820T001408Z`. `--run-id` is a label;
`YYYYMMDDTHHMMSSZ` is appended. To resume, pass the full folder name.

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
