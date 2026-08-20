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

**Full GSM8K test, no J-Lens** (`Qwen/Qwen3.5-9B`):

```bash
./scripts/uv-env run --config configs/radeon-qwen35-9b-full.yaml --evaluate
```

## NVIDIA (CUDA)

**Full GSM8K test, no J-Lens** (`Qwen/Qwen3.5-9B`):

```bash
./scripts/uv-env run --config configs/nvidia-qwen35-9b-full.yaml --evaluate
```

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
