# Terminal launchers (Qwen3.5 9B BIPIA smoke)

Shell equivalents of [`notebooks/JSpace_End_to_End_Colab.ipynb`](../notebooks/JSpace_End_to_End_Colab.ipynb) for an NVIDIA L4 host with [uv](https://docs.astral.sh/uv/).

## Quick start

```bash
export HF_TOKEN='your-hf-token'          # or: hf auth login
export OPENROUTER_API_KEY='your-key'     # required for phase2/phase4 analyze

./scripts/setup.sh
./scripts/phase1.sh
./scripts/phase2.sh
./scripts/phase3.sh
./scripts/phase4.sh
```

Default config: `configs/phase1_qwen35_9b_smoke.yaml`  
Default output: `artifacts/jspace-qwen35_9b-smoke-gpu0/`

## Scripts

| Script | Notebook | Notes |
| --- | --- | --- |
| `setup.sh` | §1 | `uv sync`, BIPIA checkout, AgentDojo/InjecAgent clones |
| `phase1.sh` | §5 | GPU; `STAGE=prepare\|capture\|analyze\|all` |
| `phase2.sh` | §7–8 | `STAGE=generate\|analyze\|all`; analyze needs OpenRouter |
| `phase3.sh` | §10 | CPU-only |
| `phase4.sh` | §11 | `STAGE=generate\|analyze\|all` |

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `JSPACE_CONFIG_PATH` | `configs/phase1_qwen35_9b_smoke.yaml` | Scientific YAML |
| `JSPACE_PHYSICAL_GPU_INDEX` | `0` | Sets `CUDA_VISIBLE_DEVICES` (use `0`–`3` on 4×L4) |
| `JSPACE_RUN_ROOT` | `artifacts/jspace-qwen35_9b-smoke-gpu{N}` | Run directory |
| `JSPACE_BENCHMARKS_ROOT` | `../jspace-benchmarks` | AgentDojo + InjecAgent parent |
| `WEBQA_TRAIN_PATH` | unset | Full-run Phase 1 only |
| `SUMMARIZATION_TRAIN_PATH` | unset | Full-run Phase 1 only |
| `STAGE` | `all` | Per-phase stage override |

## Resume examples

```bash
STAGE=capture ./scripts/phase1.sh
STAGE=generate ./scripts/phase2.sh
STAGE=analyze ./scripts/phase2.sh
JSPACE_PHYSICAL_GPU_INDEX=2 ./scripts/phase1.sh
```


## K/W experiments

Edit `sparsity_k_values`, `output_token_windows`, and `phase2.alphas` in the shared YAML. The Phase 1 launcher captures prompts once and selects one layer per K. The Phase 2 launcher automatically follows `selected_layers.json` and runs the Cartesian product of K and W, with all alphas inside each combination. A scalar legacy handoff is also supported.

Results for individual combinations are in `phase2/k{K}/w{W}/`; combined results and plots are at `phase2/`. Quality judging runs for both clean and attack outputs and uses the same OpenRouter key as attack judging. Use a new run root when changing the sweep.

Phase 3 and 4 remain single-K stages. Set `JSPACE_K=20` for both launchers to select K=20, with separate output directories/run settings for its detector artifacts. Without this variable, a sweep containing K=25 selects it; other sweeps require explicit selection.
