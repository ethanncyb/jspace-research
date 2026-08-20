# Ready-made GSM8K run configs

Self-contained YAMLs (model + host + benchmark). Run from `qwen-gsm8k-jlens/`.

First sync the matching environment:

```bash
./scripts/uv-env sync
```

## Radeon 8060S (ROCm)

**5-example smog with J-Lens capture** (`Qwen/Qwen3.5-9B-Base`):

```bash
./scripts/uv-env run --config configs/radeon-qwen35-9b-smog.yaml --evaluate
```

**500-example subset, no J-Lens** (`Qwen/Qwen3.5-9B`):

```bash
./scripts/uv-env run --config configs/radeon-qwen35-9b-500.yaml --evaluate
```

**Full GSM8K test, no J-Lens** (`Qwen/Qwen3.5-9B`):

```bash
./scripts/uv-env run --config configs/radeon-qwen35-9b-full.yaml --evaluate
```

## NVIDIA (CUDA)

NVIDIA ready-made configs default to `runtime.gpus: [0, 1]` and `runtime.parallel: true`.
Each selected GPU loads a **full model copy** and runs a disjoint shard of examples.

**5-example smog with J-Lens capture** (`Qwen/Qwen3.5-9B-Base`):

```bash
./scripts/uv-env run --config configs/nvidia-qwen35-9b-smog.yaml --evaluate
```

**500-example subset, no J-Lens** (`Qwen/Qwen3.5-9B`):

```bash
./scripts/uv-env run --config configs/nvidia-qwen35-9b-500.yaml --evaluate
```

**Full GSM8K test, no J-Lens** (`Qwen/Qwen3.5-9B`):

```bash
./scripts/uv-env run --config configs/nvidia-qwen35-9b-full.yaml --evaluate
```

### Pick GPUs

Edit `runtime.gpus` in the YAML, or override on the CLI:

```bash
# use GPUs 0 and 2
./scripts/uv-env run --config configs/nvidia-qwen35-9b-full.yaml --gpus 0,2 --evaluate

# single GPU (no data-parallel)
./scripts/uv-env run --config configs/nvidia-qwen35-9b-full.yaml --gpus 1 --evaluate
```

With one GPU, the run is sequential on that card. With two or more (and
`parallel: true`), examples are round-robin sharded across workers.

## Inspect / resume

```bash
# print resolved config (no model load)
./scripts/uv-env run python -m gsm8k_jspace inspect-config \
  --config configs/radeon-qwen35-9b-smog.yaml

# resume an existing run folder
./scripts/uv-env run --config configs/radeon-qwen35-9b-smog.yaml \
  --run-id <full_run_folder_name> --evaluate
```

Outputs land under `outputs/gsm8k/<run_id>/`.

## Host overlays

| file | profile |
|---|---|
| `configs/hosts/apple.yaml` | M1 Max / MLX |
| `configs/hosts/amd.yaml` | Radeon 8060S / ROCm |
| `configs/hosts/nvidia.yaml` | A100 / H100 / CUDA |
