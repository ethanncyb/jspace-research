# Ready-made GSM8K run configs

Self-contained YAMLs (model + host + benchmark). Run from `qwen-gsm8k-jlens/`.
Full CLI flag reference and per-benchmark examples: [COMMANDS.md](../COMMANDS.md).

Injection-suite smokes (same CLI, different `benchmark.name`):

```bash
./scripts/uv-env run --config configs/bipia-smoke.yaml --evaluate --no-capture
./scripts/uv-env run --config configs/agentdojo-smoke.yaml --evaluate --no-capture
./scripts/uv-env run --config configs/injecagent-smoke.yaml --evaluate --no-capture
```

Outputs land under `outputs/bipia/`, `outputs/agentdojo/`, and `outputs/injecagent/`.

First sync the matching environment:

```bash
./scripts/uv-env sync
```

Each size has a **basic** (accuracy only) and **jlens** (full recording) pair.
All of these set `outputs.on_existing: resume` so interrupted runs can continue.

## Capture knobs (jlens configs)

Edit these in the YAML:

| Knob | Meaning |
|---|---|
| `capture.tokens.mode` | `full_sequence` = replay after generate; `all_generated` = live during decode |
| `capture.layers.mode` | e.g. `all_fitted`, `late`, or `explicit` |
| `capture.top_k_tokens` | Top-K size for J-space / logit / model lists (default `10`) |
| `capture.fields.hidden_norm` / `jspace_norm` | layer activation norms |
| `capture.fields.top_jspace_tokens` | Top-K J-space tokens per layer × token |
| `capture.fields.hidden_vector` / `jspace_vector` | full vectors (sidecar `.vectors.pt`, off by default) |

## Radeon 8060S (ROCm)

**5-example smoke — basic:**

```bash
./scripts/uv-env run --config configs/radeon-qwen35-9b-smoke-basic.yaml --evaluate
```

**5-example smoke — full J-Lens:**

```bash
./scripts/uv-env run --config configs/radeon-qwen35-9b-smoke-jlens.yaml --evaluate
```

**500-example — basic:**

```bash
./scripts/uv-env run --config configs/radeon-qwen35-9b-500-basic.yaml --evaluate
```

**500-example — full J-Lens:**

```bash
./scripts/uv-env run --config configs/radeon-qwen35-9b-500-jlens.yaml --evaluate
```

**Full GSM8K test, no J-Lens:**

```bash
./scripts/uv-env run --config configs/radeon-qwen35-9b-full.yaml --evaluate
```

## NVIDIA (CUDA)

500-example (and full) configs default to `runtime.gpus: [0, 1]` and `runtime.parallel: true`.
Each selected GPU loads a **full model copy**.

**5-example smoke — basic:**

```bash
./scripts/uv-env run --config configs/nvidia-qwen35-9b-smoke-basic.yaml --evaluate
```

**5-example smoke — full J-Lens:**

```bash
./scripts/uv-env run --config configs/nvidia-qwen35-9b-smoke-jlens.yaml --evaluate
```

**500-example — basic:**

```bash
./scripts/uv-env run --config configs/nvidia-qwen35-9b-500-basic.yaml --evaluate
```

**500-example — full J-Lens:**

```bash
./scripts/uv-env run --config configs/nvidia-qwen35-9b-500-jlens.yaml --evaluate
```

**Full GSM8K test, no J-Lens:**

```bash
./scripts/uv-env run --config configs/nvidia-qwen35-9b-full.yaml --evaluate
```

### Pick GPUs

```bash
./scripts/uv-env run --config configs/nvidia-qwen35-9b-500-basic.yaml --gpus 0,2 --evaluate
./scripts/uv-env run --config configs/nvidia-qwen35-9b-500-basic.yaml --gpus 1 --evaluate
```

## Resume after stop / pause

Progress is written to `progress.json` after each finished example.
Completed IDs live in `completions.jsonl`. To continue:

```bash
./scripts/uv-env run --config configs/nvidia-qwen35-9b-500-basic.yaml \
  --run-id <full_run_folder_name> --evaluate
```

Inspect progress:

```bash
cat outputs/gsm8k/<run_id>/progress.json
```

## Inspect config

`uv-env run` always invokes the `run` subcommand. Use the module CLI
for `inspect-config`:

```bash
python -m gsm8k_jspace inspect-config \
  --config configs/radeon-qwen35-9b-smoke-jlens.yaml
```

Outputs land under `outputs/gsm8k/<run_id>/`.

## Host overlays

| file | profile |
|---|---|
| `configs/hosts/apple.yaml` | M1 Max / MLX |
| `configs/hosts/amd.yaml` | Radeon 8060S / ROCm |
| `configs/hosts/nvidia.yaml` | A100 / H100 / CUDA |
