# AGENTS.md

Map of this repository for coding agents. How to run: [README.md](README.md). Scientific spec: [PLAN.md](PLAN.md). Pipeline, modules, and artifacts: [docs/architecture.md](docs/architecture.md).

Read PLAN.md only when changing experimental design. Do not ingest it for ordinary code edits.

## What this is

Focused J-space prompt-injection research. Question: does prompt injection produce a reproducible representation in an LLM's J-space, and can that representation detect or causally reduce attack success?

This repo implements **Phases 1–4 only** of PLAN.md: layer selection, coarse J-space removal, two frozen detectors, then held-out / cross-benchmark transfer. Active work is on `prompt-injection-experiment`. Package: `src/jspace_research/`.

Primary model: `google/gemma-4-12B-it` (BF16, single `cuda:0`). Primary lens: `solarkyle/jspace-lenses` `gemma-4-12b-it/lens.pt`.

## Do not implement without an explicit PLAN.md revision

- Phases 5–7: recognition vs compliance, direction-specific intervention, read-only monitoring
- Plugin frameworks, base phase classes, registries, speculative extension points
- Extra models, extra probe architectures, multi-layer detectors, decomposition sweeps
- Downloading or reconstructing licensed WebQA / Summarization source datasets

`pint-benchmark/` is an unrelated local checkout. It is not part of this experiment and is not wired into the pipeline.

## Engineering conventions

From PLAN.md §2.1:

- Keep each phase's scientific logic in that phase package.
- Share code only for a concrete repeated need. Intended shared surface: [`runtime.py`](src/jspace_research/runtime.py) (atomic I/O, JSONL resume, provenance) and [`model.py`](src/jspace_research/model.py) (GPU / generation / intervention boundary).
- Frozen artifacts, identity hashes, and resumable caches are scientific safeguards. Identity mismatch is a hard failure; use a new output directory.
- Do not refactor working scientific code merely to shorten it.
- Stop when the current phase's research question and completion criteria are met.

## Frozen scientific constants

Config validators enforce these. Do not loosen them to "make a run work":

| Setting | Value |
| --- | --- |
| Seed | `42` |
| Sparsity | `k=25`, `screen_candidates=512` |
| Phase 2 alphas | `{0.0, 0.5, 1.0}` |
| Precision | BF16, no quantization |
| GPU | entire model on `cuda:0`; no CPU/disk offload |

Pinned revisions (also in YAML and provenance):

- Gemma: `5926caa4ec0cac5cbfadaf4077420520de1d5205`
- Lens repo: `1d95a2fc8a5c5a26c75a8c01c145173353e5fb65`; SHA-256 in the Phase 1 YAML
- jacobian-lens: `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- BIPIA: `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`
- AgentDojo: `089ed468cf3ed0322acc66b0211f26d9d90dbf60` / `v1.2.2`
- InjecAgent: `f19c9f2c79a41046eb13c03c51a24c567a8ffa07`

## How to work in this repo

```bash
uv sync --extra test --extra phase4
pytest
```

Tests do not need the 12B model or a GPU. Ruff: line length 100, Python 3.10, rules E/F/I/UP/B.

CLIs: `jspace-phase1`, `jspace-phase2`, `jspace-phase3`, `jspace-phase4`. Smoke config: [`configs/phase1_smoke.yaml`](configs/phase1_smoke.yaml). Full config: [`configs/phase1_full.yaml`](configs/phase1_full.yaml).

GPU stages: Phase 1 `capture`/`analyze`, Phase 2 `generate`, Phase 4 `generate`. CPU/API: Phase 2 `analyze`, Phase 3, Phase 4 `analyze` (OpenRouter only for BIPIA). Full command sequences and split-machine workflow: README.md.

Reuse the same `--output-dir` to resume an identical run. After a scientific setting, manifest, model, lens, or cache shape change, create a new directory.

## Where to edit

| Area | Path |
| --- | --- |
| Layer selection, BIPIA pairs, J-space reconstruction | `src/jspace_research/phase1/` |
| Intervention generation, ASR / utility scoring | `src/jspace_research/phase2/` |
| Mean + logistic detectors (CPU) | `src/jspace_research/phase3/` |
| Held-out BIPIA test, AgentDojo, InjecAgent | `src/jspace_research/phase4/` |
| Shared I/O and provenance | `src/jspace_research/runtime.py` |
| HuggingFace model / intervention hook | `src/jspace_research/model.py` |
| Experiment YAML | `configs/` |
| Tests (mirror phases; no GPU) | `tests/` |
| Colab launcher | `notebooks/JSpace_End_to_End_Colab.ipynb` |

Handoff: Phase 2/3/4 read frozen `selected_layer.json`. Phase 4 also reads Phase 3 `*.pt` detectors. Phase 4 does **not** consume Phase 2.

Python load:

```python
from jspace_research.phase1 import load_selected_layer
```
