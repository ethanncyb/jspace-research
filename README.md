# J-Space Prompt-Injection Research

This repository implements Phase 1 of the experiment in [`PLAN.md`](PLAN.md): select the fitted J-lens layer whose sparse J-space representation most reproducibly separates matched BIPIA prompt-injection and benign-control prompts.

The implementation deliberately stops after layer selection. It does not select a detector threshold, evaluate BIPIA test data or transfer benchmarks, generate behavioral responses, intervene on activations, or implement a gate.

## Setup

Python 3.10+ and a BF16-capable CUDA GPU are required for model capture and J-space reconstruction. The primary Gemma 4 12B run is intended for an A100-class Colab or a comparable remote CUDA host.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'

git clone https://github.com/microsoft/BIPIA.git /path/to/BIPIA
git -C /path/to/BIPIA checkout a004b69ec0dd446e0afd461d98cb5e96e120a5d0
```

Gemma and the released lens may require an authenticated Hugging Face session:

```bash
huggingface-cli login
```

## Run Phase 1

Start with the EmailQA smoke configuration:

```bash
jspace-phase1 \
  --config configs/phase1_smoke.yaml \
  --bipia-root /path/to/BIPIA/benchmark \
  --output-dir /path/to/output \
  --stage all
```

The stages may be run separately and safely resumed:

```bash
jspace-phase1 --config configs/phase1_smoke.yaml --bipia-root /path/to/BIPIA/benchmark --output-dir /path/to/output --stage prepare
jspace-phase1 --config configs/phase1_smoke.yaml --bipia-root /path/to/BIPIA/benchmark --output-dir /path/to/output --stage capture
jspace-phase1 --config configs/phase1_smoke.yaml --bipia-root /path/to/BIPIA/benchmark --output-dir /path/to/output --stage analyze
```

For the full five-task run, provide researcher-created BIPIA-format WebQA and Summarization training files:

```bash
jspace-phase1 \
  --config configs/phase1_full.yaml \
  --bipia-root /path/to/BIPIA/benchmark \
  --webqa-train /path/to/webqa/train.jsonl \
  --summarization-train /path/to/summarization/train.jsonl \
  --output-dir /path/to/output \
  --stage all
```

The Colab entry point is [`notebooks/JSpace_Layer_Selection_Colab.ipynb`](notebooks/JSpace_Layer_Selection_Colab.ipynb).

## Method and outputs

At each fitted layer, the pipeline constructs normalized token directions from rows of `W_U @ J_l`. It reconstructs the final-prompt-token residual as a sparse nonnegative combination using a screened greedy approximation: 512 positive candidates, at most 25 selected atoms, and an iterative nonnegative support refit. This is an approximation, not an exact orthogonal projection and not Anthropic's exact gradient-pursuit implementation.

The output directory contains:

- `pair_manifest.jsonl`: frozen train/validation attack-control pairs;
- `provenance.json`: lightweight fixed-input and runtime provenance;
- resumable activation and per-layer J-space caches;
- per-layer direction and validation-score artifacts;
- `layer_metrics.csv` and `validation_scores.parquet`;
- `selected_layer.json` and `selected_layer_direction.pt`;
- the macro-AUPRC and selected-layer score-distribution plots.

If configuration, manifest, model, lens, layer, or cache-shape identity changes, the pipeline stops rather than silently reusing stale cached data. Use a new output directory for a different run.
