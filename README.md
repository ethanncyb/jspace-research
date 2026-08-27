# J-Space Prompt-Injection Research

This repository implements Phase 1 of the experiment in [`PLAN.md`](PLAN.md): select the fitted J-lens layer whose sparse J-space representation most reproducibly separates matched BIPIA prompt-injection and benign-control prompts.

The implementation deliberately stops after layer selection. It does not select a detector threshold, evaluate BIPIA test data or transfer benchmarks, generate behavioral responses, intervene on activations, or implement a gate.

## Where to run it

Model capture and J-space reconstruction require Python 3.10+, a BF16-capable NVIDIA CUDA GPU, and enough storage for the model and resumable caches. An A100-class GPU is recommended for the primary Gemma 4 12B run.

There are two supported GPU workflows:

| Option | Execution location | Persistent output |
| --- | --- | --- |
| Google Colab | A hosted Colab GPU runtime | Mount Google Drive or download the run directory before the runtime ends |
| Cloud GPU over SSH | A CUDA server reached from a local terminal | Use the server's persistent disk, then copy the run directory locally with `rsync` or `scp` |

A typical Mac can run dataset preparation, tests, metrics, and other lightweight CPU work, but it cannot run the current `capture` or `analyze` stages because those stages require CUDA. The experiment output is file-based, so Phase 1 may run on either GPU option and its completed run directory may then be used on another machine.

## Shared requirements

The repository pins:

- Jacobian-lens to `581d398613e5602a5af361e1c34d3a92ea82ba8e`;
- BIPIA to `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`;
- the model and fitted lens to the revisions and hashes in the Phase 1 YAML configurations.

Gemma and the released lens may require accepting their licenses and authenticating with Hugging Face. The smoke run needs only the pinned BIPIA checkout. The full run additionally requires researcher-provided WebQA and Summarization files in BIPIA `train.jsonl` format. In accordance with the experiment plan, the pipeline does not download or reconstruct those licensed source datasets.

Always start with the smoke run. It uses EmailQA, 12 training pairs, 6 validation pairs, and six fitted layers. It verifies the pipeline but is not the final scientific experiment. The full configuration uses all five tasks and all fitted layers.

## Option A: Google Colab

Open [`notebooks/JSpace_Layer_Selection_Colab.ipynb`](notebooks/JSpace_Layer_Selection_Colab.ipynb), or [launch it directly in Colab](https://colab.research.google.com/github/ethanncyb/jspace-research/blob/prompt-injection-experiment/notebooks/JSpace_Layer_Selection_Colab.ipynb).

1. In Colab, select **Runtime → Change runtime type → GPU**. An A100-class runtime is recommended.
2. Run the installation cell. It clones the `prompt-injection-experiment` branch, installs the package, checks out pinned BIPIA, and prints the resolved repository revisions.
3. Authenticate when `notebook_login()` prompts. Confirm that the Hugging Face account has access to the pinned Gemma model and lens.
4. Leave `RUN_MODE = "smoke"` for the first run. Use `RUN_MODE = "full"` only after smoke succeeds and the two external task files are available.
5. Run Phase 1 and inspect the displayed selection result, metrics, and plots.

Confirm the selected runtime before starting the expensive stages:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

The pipeline also refuses to run `capture` or `analyze` without CUDA and records the GPU identity in `provenance.json`.

### Persist Colab results

Paths under `/content` belong to the temporary Colab VM. To retain the complete run, mount Google Drive before configuring `OUTPUT_DIR`:

```python
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

RUN_MODE = "smoke"  # change to "full" for the scientific run
RUN_NAME = "phase1-smoke-run-001" if RUN_MODE == "smoke" else "phase1-full-run-001"
OUTPUT_DIR = Path("/content/drive/MyDrive/jspace-research/runs") / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

For a full run, place the external datasets in persistent storage as well and configure their paths:

```python
WEBQA_TRAIN_PATH = Path("/content/drive/MyDrive/jspace-research/data/webqa/train.jsonl")
SUMMARIZATION_TRAIN_PATH = Path(
    "/content/drive/MyDrive/jspace-research/data/summarization/train.jsonl"
)
```

Writing directly to Drive prioritizes persistence over I/O speed. Alternatively, run under `/content` and copy the entire output directory to Drive before the runtime ends. A fresh Colab runtime is recommended after the research branch changes because the installation cell reuses an existing checkout within the same runtime.

## Option B: Cloud GPU over SSH

SSH provides a terminal on the remote GPU server; it does not automatically share the laptop's filesystem. Use a persistent server volume for the repository, datasets, model cache, and experiment output.

From the remote shell, clone and install the experiment:

```bash
git clone --branch prompt-injection-experiment \
  https://github.com/ethanncyb/jspace-research.git
cd jspace-research

python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'

git clone https://github.com/microsoft/BIPIA.git /path/to/BIPIA
git -C /path/to/BIPIA checkout a004b69ec0dd446e0afd461d98cb5e96e120a5d0
hf auth login
```

Verify CUDA before running:

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Run the smoke configuration first:

```bash
jspace-phase1 \
  --config configs/phase1_smoke.yaml \
  --bipia-root /path/to/BIPIA/benchmark \
  --output-dir ./artifacts/phase1-smoke-run-001 \
  --stage all
```

Then run the full five-task experiment:

```bash
jspace-phase1 \
  --config configs/phase1_full.yaml \
  --bipia-root /path/to/BIPIA/benchmark \
  --webqa-train /path/to/webqa/train.jsonl \
  --summarization-train /path/to/summarization/train.jsonl \
  --output-dir ./artifacts/phase1-full-run-001 \
  --stage all
```

After a completed stage or run, copy the output to the local repository from a terminal on the laptop:

```bash
rsync -av --partial \
  user@gpu-server:/remote/path/jspace-research/artifacts/phase1-full-run-001/ \
  /local/path/jspace-research/artifacts/phase1-full-run-001/
```

Replace the example user, host, and paths with those supplied by the GPU provider. Do not rely on an instance's temporary boot disk for the only copy of a run.

## Stages and resumption

`--stage all` runs `prepare`, `capture`, and `analyze` in sequence. They may instead be run separately with the same configuration, inputs, and output directory:

```bash
jspace-phase1 --config configs/phase1_smoke.yaml --bipia-root /path/to/BIPIA/benchmark --output-dir /path/to/output --stage prepare
jspace-phase1 --config configs/phase1_smoke.yaml --bipia-root /path/to/BIPIA/benchmark --output-dir /path/to/output --stage capture
jspace-phase1 --config configs/phase1_smoke.yaml --bipia-root /path/to/BIPIA/benchmark --output-dir /path/to/output --stage analyze
```

The activation and decomposition caches are resumable. Reuse the exact same output directory to resume the same run. Scientific identity excludes machine-local dataset and output paths, so a complete run directory can be moved between machines without changing its identity. The source datasets are required for `prepare`; after the manifest is frozen, `capture` and `analyze` read the manifest and do not require the original source files. If a scientific setting, manifest, model, lens, layer, or cache shape changes, the pipeline stops rather than silently reusing stale data; create a new output directory for a different run.

## Phase 1 output and handoff

At each fitted layer, the pipeline constructs normalized token directions from rows of `W_U @ J_l`. It reconstructs the final-prompt-token residual as a sparse nonnegative combination using a screened greedy approximation: 512 positive candidates, at most 25 selected atoms, and an iterative nonnegative support refit. This is an approximation, not an exact orthogonal projection and not Anthropic's exact gradient-pursuit implementation.

The output directory contains:

- `pair_manifest.jsonl`: frozen train/validation attack-control pairs;
- `provenance.json`: lightweight fixed-input and runtime provenance;
- resumable activation and per-layer J-space reconstruction caches;
- sparse support-ID and coefficient caches for downstream detector training;
- per-layer direction and validation-score artifacts;
- `layer_metrics.csv` and `validation_scores.parquet`;
- `selected_layer.json` and `selected_layer_direction.pt`;
- the macro-AUPRC and selected-layer score-distribution plots.

Keep the complete full-run directory for auditability and resumption. The revised Phase 2 performs coarse removal of the selected layer's reconstructed J-space component, so its handoff includes the selected-layer activation and decomposition caches—not only the selected direction. `selected_layer.json` records the frozen run identity, direction hash, selected-layer cache locations, decomposition settings, and relative artifact paths so later phases can load the handoff without notebook state. Phase 3 can reuse the saved sparse support IDs and coefficients without repeating J-space reconstruction.

Verify and load a copied handoff with:

```python
from jspace_research.phase1 import load_selected_layer

selection, direction = load_selected_layer(
    "artifacts/phase1-full-run-001/selected_layer.json"
)
print(selection["run_id"], selection["selected_layer"])
```

When moving between Colab, a cloud GPU, and a local machine, preserve the directory structure. A conventional local destination is:

```text
jspace-research/artifacts/phase1-full-run-001/
```

The repository ignores `artifacts/`, generated outputs, and caches so experimental data is not accidentally committed.

## Development checks

The automated tests do not require the 12B model or a GPU:

```bash
pip install -e '.[test]'
pytest
```
