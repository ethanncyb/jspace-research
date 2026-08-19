# 09 — Python implementation and Jupyter visualization

## Technology decision

The benchmark, capture, intervention, evaluation, comparison, and artifact
layers are implemented in Python.

Jupyter notebooks are the primary human-readable analysis interface. They load
saved artifacts, call reusable Python analysis functions, render tables and
figures, and can be saved as executed `.ipynb` files and standalone HTML.

Notebooks do not:

- load the 9B model by default;
- generate benchmark completions;
- modify J-Space activations;
- contain the only implementation of a metric;
- overwrite raw run artifacts.

This separation keeps experiments resumable and makes every visualization
reproducible from an immutable run directory.

## Proposed project additions

```text
qwen-gsm8k-jlens/
├── notebooks/
│   ├── 00-run-overview.ipynb
│   ├── 01-gsm8k-accuracy.ipynb
│   ├── 02-jspace-capture.ipynb
│   ├── 03-intervention-comparison.ipynb
│   ├── 04-cross-hardware-replication.ipynb
│   └── 05-example-explorer.ipynb
├── src/gsm8k_jspace/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── tables.py
│   │   ├── statistics.py
│   │   └── plots.py
│   └── ...
├── configs/
│   └── visualization.yaml
└── outputs/
    └── <run_id>/
        └── visualization/
            ├── executed/
            │   └── <notebook>.ipynb
            ├── html/
            │   └── <notebook>.html
            ├── figures/
            │   ├── *.png
            │   └── *.svg
            └── tables/
                └── *.csv
```

Source notebooks are clean templates committed to git. Executed notebooks and
exports belong under the relevant run/comparison output directory.

## Python analysis API

Notebook cells should be short and declarative:

```python
from gsm8k_jspace.analysis import load_run, plot_accuracy_summary

run = load_run(RUN_DIR)
plot_accuracy_summary(run)
```

The package owns:

- schema-version checks;
- streaming/compressed JSONL and Parquet loading;
- answer/capture/comparison joins;
- aggregation and statistical calculations;
- plotting functions;
- consistent labels, ordering, and export behavior.

This avoids duplicating parsing or statistics across notebooks.

## Notebook parameters

Every notebook begins with one tagged parameter cell:

```python
RUN_DIR = "../outputs/gsm8k/example-run"
BASELINE_DIR = None
CANDIDATE_DIR = None
OUTPUT_DIR = None
MAX_EXAMPLES = 100
```

Paths are parameters rather than embedded absolute host paths. Automated
execution overrides them and records the resolved values in the notebook.

Notebooks validate manifests before displaying results. A notebook must show a
visible warning when:

- the run is incomplete;
- baseline and candidate manifests are incompatible;
- environments or hardware differ;
- extraction failures are present;
- an identity J-Lens or CPU fallback was used;
- capture data is missing or truncated.

## Notebook responsibilities

### `00-run-overview.ipynb`

Purpose: quickly understand one run.

Displays:

- model, dataset, condition, prompt, generation, and capture configuration;
- uv environment, backend, hardware, dtype, and package versions;
- run status, completion count, timing, and artifact integrity;
- selected layers and token policy;
- warnings and compatibility status.

### `01-gsm8k-accuracy.ipynb`

Purpose: inspect task performance.

Displays:

- exact-answer accuracy and confidence interval;
- extraction success/failure rate;
- correct/incorrect counts;
- generated token and latency distributions;
- answer parser method counts;
- sortable failed-example table;
- selected example prompt, completion, gold answer, and parsed prediction.

### `02-jspace-capture.ipynb`

Purpose: inspect observation-only J-Space data.

Displays:

- hidden and J-Space norms by layer;
- layer × generated-position heatmaps;
- trajectories for selected examples;
- top J-Space vocabulary readouts at selected tokens;
- word-end versus model-token views;
- capture row/file size and integrity summaries.

The notebook streams or filters capture files before converting to pandas so a
full all-token run does not exhaust notebook memory.

### `03-intervention-comparison.ipynb`

Purpose: compare a same-hardware baseline and intervention.

Displays:

- both accuracies and paired difference;
- exact McNemar result and paired bootstrap interval;
- correct-both, incorrect-both, broken, and fixed counts;
- per-layer hidden/J-Space perturbation magnitude;
- strength-response plots;
- broken/fixed example explorer;
- compatibility warnings before any causal interpretation.

### `04-cross-hardware-replication.ipynb`

Purpose: summarize M1 Max, Radeon 8060S, A100, and H100 runs.

Displays:

- backend/environment table;
- small/medium/large run coverage;
- accuracy and extraction rate by hardware;
- completion agreement for shared example IDs;
- runtime, throughput, and peak memory;
- explicit separation between same-hardware paired results and cross-hardware
  replication summaries.

It must not pair an M1 baseline with a Radeon or NVIDIA intervention.

### `05-example-explorer.ipynb`

Purpose: inspect one GSM8K example deeply.

Displays:

- question, prompt, rationale, gold answer, and predicted answer;
- baseline/intervention generated text side by side;
- tokenized output with absolute/generated/word indices;
- selected layer × token J-Space norms and readouts;
- intervention deltas;
- top-k readouts at the last token or selected word.

## Visualization standards

Use a small common plotting layer with:

- pandas for tables;
- matplotlib and seaborn for deterministic static charts;
- optional Plotly only where interaction materially helps exploration;
- consistent condition colors and layer ordering;
- readable titles including run IDs and conditions;
- axis labels with units;
- confidence intervals where applicable;
- colorblind-readable palettes;
- no manually entered result values.

Every plot function accepts an output path. Publication/report figures are
saved as PNG and SVG; tabular data behind each chart is saved as CSV.

## Large-data handling

Full J-Space captures can be much larger than notebook memory. Analysis loaders
must support:

- requested columns only;
- example/layer/token filtering before materialization;
- chunked JSONL reading;
- Parquet predicate pushdown when Parquet is configured;
- bounded top-k example selection;
- cached aggregate tables keyed by source checksums.

The notebook displays how many source rows were loaded and whether sampling or
filtering was applied.

Raw full vectors are never expanded into a pandas cell per dimension. Vector
analysis loads only requested examples/layers and computes aggregates with
NumPy or PyTorch.

## uv and Jupyter environments

Jupyter belongs in an `analysis` dependency group so accelerator hosts do not
need notebook packages for unattended benchmark runs.

Expected commands:

```bash
./scripts/uv-env sync --group analysis
./scripts/uv-env jupyter
```

The launcher starts:

```bash
uv run --group analysis jupyter lab
```

from the already selected `.venv-mps`, `.venv-rocm`, `.venv-cuda`, or
`.venv-cpu` environment. This guarantees notebook imports use the same locked
project code and schemas as the benchmark.

For lightweight local analysis, `.venv-cpu` can read artifacts copied from
Radeon/A100/H100 runs; loading results does not require that run's accelerator.
The original environment remains visible through `environment.json`.

## Automated execution and saving

Add an export command:

```text
python -m gsm8k_jspace visualize \
    --notebook intervention-comparison \
    --baseline <baseline_run> \
    --candidate <intervention_run> \
    --format ipynb,html
```

The command:

1. copies the clean template to a temporary/output location;
2. injects run parameters;
3. executes it with a timeout and the selected uv kernel;
4. fails if a cell errors;
5. saves the executed notebook;
6. exports standalone HTML;
7. saves generated figures/tables;
8. records notebook source checksum, analysis package version, and input
   artifact checksums.

Use Jupyter's supported execution/export APIs at implementation time. Avoid
custom editing of notebook JSON.

## Reproducibility metadata

Each visualization export writes `visualization_manifest.json`:

```json
{
  "schema_version": 1,
  "template": "03-intervention-comparison.ipynb",
  "template_sha256": "sha256:...",
  "input_runs": ["baseline-run", "intervention-run"],
  "input_manifest_sha256": ["sha256:...", "sha256:..."],
  "uv_lock_sha256": "sha256:...",
  "analysis_package_version": "0.1.0",
  "executed_at": "2026-08-19T23:00:00Z",
  "exports": {
    "ipynb": "executed/03-intervention-comparison.ipynb",
    "html": "html/03-intervention-comparison.html"
  }
}
```

## Notebook testing

Fast tests:

- analysis loaders against fixture artifacts;
- schema and compatibility warnings;
- plot functions return figures and save files;
- aggregate tables match evaluator summaries;
- filtering/chunking returns expected rows.

Notebook tests:

- every source notebook parses as valid notebook JSON;
- parameter cells exist and are tagged;
- execute every notebook against tiny fixture runs;
- fail on cell errors or stale expected output schemas;
- verify HTML, figures, and tables are produced.

Full-run notebook tests do not rerun the model.

## Acceptance criteria

- All production logic is in importable Python modules.
- Notebooks contain presentation and orchestration, not unique metric logic.
- A user can open one notebook and understand run settings and warnings.
- Baseline/intervention results are visualized only after manifest validation.
- Large capture data is filtered or streamed safely.
- Executed notebooks, HTML, figures, and backing CSV tables can be saved.
- Notebook exports record source and input checksums.
- The same notebook templates work for M1, Radeon, A100, and H100 artifacts.
- Tiny fixture execution is automated and does not require a GPU.
