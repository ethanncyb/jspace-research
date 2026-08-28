# J-Space Prompt-Injection Research

This repository implements Phases 1–3 of the experiment in [`PLAN.md`](PLAN.md): select the fitted J-lens layer whose sparse J-space representation most reproducibly separates matched BIPIA attack/control prompts, measure behavior while removing that reconstructed component, then construct and freeze the two planned linear detectors.

The implementation deliberately stops after freezing the Phase 3 detectors and development thresholds. It does not evaluate BIPIA test data or transfer benchmarks, perform direction-specific interventions, or implement a gate.

## Where to run it

Model capture, J-space reconstruction, and intervention generation require Python 3.10+, a BF16-capable NVIDIA CUDA GPU, and enough storage for the model and resumable caches. An A100-class GPU is recommended for the primary Gemma 4 12B run. Phase 2 aggregation and all of Phase 3 run on CPU from saved artifacts.

There are three supported GPU workflows using the same commands:

| Option | Execution location | Persistent output |
| --- | --- | --- |
| Google Colab | A hosted Colab GPU runtime | Mount Google Drive or download the run directory before the runtime ends |
| Cloud GPU over SSH | A CUDA server reached from a local terminal | Use the server's persistent disk, then copy the run directory locally with `rsync` or `scp` |
| Local CUDA machine | A workstation with a compatible NVIDIA GPU | Write directly under the local repository or another persistent disk |

A typical Mac can run dataset preparation, tests, Phase 2 analysis, and all of Phase 3, but it cannot run Phase 1 `capture`/`analyze` or Phase 2 `generate`, which require CUDA. The experiment is file-based, so GPU work may run in Colab or over SSH and the complete run directory can then be copied elsewhere.

## Shared requirements

The repository pins:

- Jacobian-lens to `581d398613e5602a5af361e1c34d3a92ea82ba8e`;
- BIPIA to `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`;
- the model and fitted lens to the revisions and hashes in the Phase 1 YAML configurations.

Gemma and the released lens may require accepting their licenses and authenticating with Hugging Face. Phase 2 attack scoring also requires an `OPENAI_API_KEY`; it uses the pinned `gpt-4.1-mini-2025-04-14` judge and stores no credential in artifacts. The smoke run needs only the pinned BIPIA checkout. The full run additionally requires researcher-provided WebQA and Summarization files in BIPIA `train.jsonl` format. In accordance with the experiment plan, the pipeline does not download or reconstruct those licensed source datasets.

Always start with the end-to-end smoke run. It uses EmailQA, 12 training pairs, 6 validation pairs, six fitted layers, and the three Phase 2 conditions: intact (`0.0`), partial removal (`0.5`), and full removal (`1.0`). It also requires exact token equality between ordinary no-hook generation and the zero-strength hook. It verifies the pipeline but is not the final scientific experiment. The full configuration uses all five tasks and all fitted layers.

## Option A: Google Colab

Open the canonical [`notebooks/JSpace_End_to_End_Colab.ipynb`](notebooks/JSpace_End_to_End_Colab.ipynb), or [launch it directly in Colab](https://colab.research.google.com/github/ethanncyb/jspace-research/blob/prompt-injection-experiment/notebooks/JSpace_End_to_End_Colab.ipynb).

1. In Colab, select **Runtime → Change runtime type → GPU**. An A100-class runtime is recommended.
2. Run the installation cell. It clones the `prompt-injection-experiment` branch, installs the package, checks out pinned BIPIA, and prints the resolved revisions.
3. Authenticate with Hugging Face and add `OPENAI_API_KEY` to Colab Secrets.
4. Leave `RUN_MODE = "smoke"` for the first run. Use `RUN_MODE = "full"` only after smoke succeeds and the two external task files are available.
5. Run and inspect Phase 1, run the separate Phase 2 generation and analysis cells, then run the CPU-only Phase 3 cell.

Confirm the selected runtime before starting the expensive stages:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

The pipeline refuses to run Phase 1 `capture`/`analyze` or Phase 2 `generate` without CUDA and records GPU identity in each phase's `provenance.json`.

### Persist Colab results

Paths under `/content` belong to the temporary Colab VM. To retain the complete run, mount Google Drive before configuring `RUN_ROOT`:

```python
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

RUN_MODE = "smoke"  # change to "full" for the scientific run
RUN_NAME = f"jspace-e2e-{RUN_MODE}"
RUN_ROOT = Path("/content/drive/MyDrive/jspace-research/runs") / RUN_NAME
RUN_ROOT.mkdir(parents=True, exist_ok=True)
```

For a full run, place the external datasets in persistent storage as well and configure their paths:

```python
WEBQA_TRAIN_PATH = Path("/content/drive/MyDrive/jspace-research/data/webqa/train.jsonl")
SUMMARIZATION_TRAIN_PATH = Path(
    "/content/drive/MyDrive/jspace-research/data/summarization/train.jsonl"
)
```

Writing directly to Drive prioritizes persistence over I/O speed. Alternatively, run under `/content` and copy the entire run root to Drive before the runtime ends. Use a new run root after changing frozen scientific inputs; reusing the same root resumes an identical run.

## Option B: Local CUDA or cloud GPU over SSH

Run the following commands directly on a local CUDA workstation or in an SSH session on a cloud GPU. SSH does not automatically share the laptop's filesystem, so remote runs should use a persistent server volume for the repository, datasets, model cache, and output.

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
export OPENAI_API_KEY='your-api-key'
```

Verify CUDA before running:

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Run the smoke experiment phase by phase. Phase 1 and Phase 2 resume compatible work in the same directory; Phase 3 is a short CPU computation that safely rewrites its final artifacts:

```bash
jspace-phase1 \
  --config configs/phase1_smoke.yaml \
  --bipia-root /path/to/BIPIA/benchmark \
  --output-dir ./artifacts/smoke/phase1 \
  --stage all

jspace-phase2 \
  --config configs/phase1_smoke.yaml \
  --phase1 ./artifacts/smoke/phase1/selected_layer.json \
  --output-dir ./artifacts/smoke/phase2 \
  --stage generate

jspace-phase2 \
  --config configs/phase1_smoke.yaml \
  --phase1 ./artifacts/smoke/phase1/selected_layer.json \
  --output-dir ./artifacts/smoke/phase2 \
  --stage analyze

jspace-phase3 \
  --config configs/phase1_smoke.yaml \
  --phase1 ./artifacts/smoke/phase1/selected_layer.json \
  --output-dir ./artifacts/smoke/phase3
```

Then run the full five-task experiment:

```bash
jspace-phase1 \
  --config configs/phase1_full.yaml \
  --bipia-root /path/to/BIPIA/benchmark \
  --webqa-train /path/to/webqa/train.jsonl \
  --summarization-train /path/to/summarization/train.jsonl \
  --output-dir ./artifacts/full/phase1 \
  --stage all

jspace-phase2 \
  --config configs/phase1_full.yaml \
  --phase1 ./artifacts/full/phase1/selected_layer.json \
  --output-dir ./artifacts/full/phase2 \
  --stage generate

jspace-phase2 \
  --config configs/phase1_full.yaml \
  --phase1 ./artifacts/full/phase1/selected_layer.json \
  --output-dir ./artifacts/full/phase2 \
  --stage analyze

jspace-phase3 \
  --config configs/phase1_full.yaml \
  --phase1 ./artifacts/full/phase1/selected_layer.json \
  --output-dir ./artifacts/full/phase3
```

After a completed stage or run, copy the output to the local repository from a terminal on the laptop:

```bash
rsync -av --partial \
  user@gpu-server:/remote/path/jspace-research/artifacts/full/ \
  /local/path/jspace-research/artifacts/full/
```

Replace the example user, host, and paths with those supplied by the GPU provider. Do not rely on an instance's temporary boot disk for the only copy of a run.

## Stages, resumption, and split-machine analysis

`--stage all` runs `prepare`, `capture`, and `analyze` in sequence. They may instead be run separately with the same configuration, inputs, and output directory:

```bash
jspace-phase1 --config configs/phase1_smoke.yaml --bipia-root /path/to/BIPIA/benchmark --output-dir /path/to/output --stage prepare
jspace-phase1 --config configs/phase1_smoke.yaml --bipia-root /path/to/BIPIA/benchmark --output-dir /path/to/output --stage capture
jspace-phase1 --config configs/phase1_smoke.yaml --bipia-root /path/to/BIPIA/benchmark --output-dir /path/to/output --stage analyze
```

The activation and decomposition caches are resumable. Reuse the exact same output directory to resume the same run. Scientific identity excludes machine-local dataset and output paths, so a complete run directory can be moved between machines without changing its identity. The source datasets are required for `prepare`; after the manifest is frozen, `capture` and `analyze` read the manifest and do not require the original source files. If a scientific setting, manifest, model, lens, layer, or cache shape changes, the pipeline stops rather than silently reusing stale data; create a new output directory for a different run.

Phase 2 reads the frozen Phase 1 result directly from the same run root:

```bash
jspace-phase2 \
  --config configs/phase1_full.yaml \
  --phase1 artifacts/full/phase1/selected_layer.json \
  --output-dir artifacts/full/phase2 \
  --stage generate
```

`generate` requires Gemma to fit on one CUDA GPU. It appends and flushes each completed example/condition/alpha result to `generations.jsonl`; it never silently offloads model layers to CPU or disk. After copying the complete run root to another machine, run CPU/API scoring without loading Gemma:

```bash
export OPENAI_API_KEY='your-api-key'
jspace-phase2 \
  --config configs/phase1_full.yaml \
  --phase1 artifacts/full/phase1/selected_layer.json \
  --output-dir artifacts/full/phase2 \
  --stage analyze
```

`--stage all` runs both stages. Generation and judgments resume by job ID from `generations.jsonl` and `judgments.jsonl`. A truncated final record from an interrupted append is discarded and recomputed; malformed earlier records or mismatched config, parent handoff, prompt, generation, judge model, or rubric cause a hard failure.

Phase 3 independently reads the frozen Phase 1 handoff. It is one short CPU command with no stage or training cache, and it does not require Gemma, the lens, BIPIA source files, Phase 2, or an API key:

```bash
jspace-phase3 \
  --config configs/phase1_full.yaml \
  --phase1 artifacts/full/phase1/selected_layer.json \
  --output-dir artifacts/full/phase3
```

## Outputs and phase boundaries

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

Keep the complete full-run directory for auditability and resumption. Phase 2 receives the path to `phase1/selected_layer.json`; no manual conversion or notebook-state transfer is required. That file records the frozen run identity, direction hash, selected-layer cache locations, decomposition settings, and relative artifact paths. Phase 2 validates those artifacts before use, and Phase 3 can reuse the saved sparse support IDs and coefficients without repeating J-space reconstruction.

Phase 2 reads only that frozen handoff. Its directory contains:

- resumable append-only `generations.jsonl` and `judgments.jsonl` caches;
- `phase2_results.parquet` with baseline/current generations and per-example outcomes;
- `phase2_summary.csv` with ASR, delta-ASR, per-task utility, retention, and refusals;
- `phase2_asr_vs_alpha.png` and `phase2_clean_utility_vs_alpha.png`;
- `phase2_examples.csv` for deterministic endpoint inspection;
- lightweight `provenance.json` linking the results to the Phase 1 run, judge rubric, model pins, and generation GPU.

Clean utility is ROUGE-based reference overlap against the frozen BIPIA `construct_response` target. It is not described as a BIPIA-native task metric, and incompatible task utilities are never averaged. BIPIA's packaged ASR evaluators do not cover the required training-attack variants, so attack prompts use the fixed, provenance-recorded semantic YES/NO/UNKNOWN judge rubric described above. BIPIA defines no formal valid-output structure for these development tasks, so validity fields are null. A Phase 2 effect establishes functional involvement of the removed J-space component, not injection-specific causality.

Phase 3 writes `mean_detector.pt`, `logistic_detector.pt`, `phase3_validation_scores.parquet`, `phase3_metrics.csv`, `phase3_detector_comparison.png`, and lightweight `provenance.json`. These are development results: the mean detector is reused from Phase 1, the logistic detector is fitted only on Phase 1 training examples, and both thresholds are selected on Phase 1 validation examples. Phase 4 is responsible for unbiased held-out evaluation.

Verify and load a copied handoff with:

```python
from jspace_research.phase1 import load_selected_layer

selection, direction = load_selected_layer(
    "artifacts/full/phase1/selected_layer.json"
)
print(selection["run_id"], selection["selected_layer"])
```

When moving between Colab, a cloud GPU, and a local machine, preserve the directory structure. A conventional local destination is:

```text
jspace-research/artifacts/full/
```

The repository ignores `artifacts/`, generated outputs, and caches so experimental data is not accidentally committed.

## Development checks

The automated tests do not require the 12B model or a GPU:

```bash
pip install -e '.[test]'
pytest
```
