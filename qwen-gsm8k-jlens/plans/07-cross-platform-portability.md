# 07 — Cross-platform portability

## Supported execution targets

The implementation must treat hardware as a runtime capability, not as a
hard-coded CUDA assumption.

| target | PyTorch backend | initial support level | notes |
|---|---|---|---|
| Apple Silicon M1/MPS | `mps` | supported | macOS; float16 is the safe default |
| NVIDIA GPU | CUDA | supported | Linux or Windows with compatible PyTorch/CUDA |
| AMD Radeon 8060S | ROCm/HIP | supported on validated ROCm OS | PyTorch exposes ROCm through much of the `torch.cuda` API |
| CPU | CPU | correctness/smoke support | too slow for a normal 9B full run |

AMD's current ROCm compatibility matrix lists Radeon 8060S. Exact operating
system, driver, ROCm, Python, and PyTorch versions must still follow AMD's
matrix at installation time:

- [AMD ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [AMD PyTorch installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)

ROCm support should be validated first on Linux using an official AMD wheel or
container. A Radeon device on an unsupported Windows/driver combination is not
equivalent to a validated ROCm environment; DirectML is outside the initial
support scope.

## Memory feasibility

“Supported platform” does not mean every machine can hold every configured
model.

Qwen 3.5 9B requires approximately 18 GB for 16-bit model weights alone.
Additional memory is needed for:

- the loaded J-Lens matrices;
- pseudo-inverses for intervention layers;
- KV cache during generation;
- temporary hidden/J-Space tensors;
- optional full-vector captures;
- framework and device runtime overhead.

Consequences:

- an 8 GB or 16 GB M1 MacBook cannot run the exact 9B + fitted J-Lens
  experiment entirely on MPS;
- higher-memory Apple Silicon machines may run it with conservative generation,
  layer, and capture settings;
- CPU/disk offload may make a run possible but much slower and must be recorded
  as a distinct execution environment;
- a smaller model is valid only when a matching fitted J-Lens exists; an
  identity lens is a plumbing control, not a scientific substitute.

The program must perform a preflight memory estimate and warn or fail before
loading when the selected configuration is clearly infeasible.

## Backend abstraction

Add a `platform/` package:

```text
src/gsm8k_jspace/platform/
├── __init__.py
├── capabilities.py
├── device.py
├── memory.py
└── diagnostics.py
```

It owns:

- backend detection;
- device and dtype selection;
- supported-operation probes;
- memory estimates;
- synchronization and memory-stat helpers;
- normalized environment metadata.

No other module may:

- call `.cuda()` directly;
- construct tensors on a hard-coded device;
- assume `torch.cuda.is_available()` means NVIDIA;
- assume model, hidden state, J-Lens, and pseudo-inverse share a device;
- use CUDA-only autocast or memory APIs without capability checks.

## Reliable backend detection

Detection order:

1. explicit configured backend/device;
2. ROCm when `torch.version.hip` is present and `torch.cuda.is_available()`;
3. NVIDIA CUDA when `torch.version.cuda` is present and CUDA is available;
4. MPS when built and available;
5. CPU.

ROCm commonly uses the `torch.cuda` Python namespace. Therefore metadata and
logs must distinguish:

```python
if torch.version.hip:
    backend = "rocm"
elif torch.version.cuda:
    backend = "cuda"
```

The resolved device must be based on actual model/tensor placement when
`device_map: auto` or offload is active.

## Dtype policy

`dtype: auto` resolves by tested capability:

| backend | preferred | fallback |
|---|---|---|
| CUDA | bfloat16 when supported | float16, then float32 |
| ROCm | bfloat16 when supported | float16, then float32 |
| MPS/M1 | float16 | float32 |
| CPU | float32 | bfloat16 only after an explicit operation probe |

The implementation must run a small operation probe for matrix multiplication,
unembedding, norm, top-k, and pseudo-inverse/back-projection before a real run.
Unsupported requested dtypes fail with guidance; they are not silently changed
after the manifest is created.

## Pseudo-inverse policy

`torch.linalg.pinv` is expensive and backend support/performance varies.
Configuration must separate model execution from J-Lens linear algebra:

```yaml
runtime:
  backend: "auto"                    # auto | mps | cuda | rocm | cpu
  device: "auto"
  linear_algebra_device: "auto"      # auto | model | cpu
  pinv:
    compute_device: "cpu"
    cache_device: "cpu"
    cache_dir: ".cache/jspace/pinv"
    preload: false
```

Recommended policy:

- compute pseudo-inverses in float32 on CPU by default;
- cache by lens checksum, layer, dtype, and algorithm version;
- move only the needed back-projection matrix/delta to the operation device;
- never preload pseudo-inverses for all layers on low-memory systems;
- permit accelerator computation only after the capability probe passes.

This policy avoids relying on MPS support for every linear algebra operation
and reduces accelerator memory pressure across MPS, CUDA, and ROCm.

## Hugging Face loading policy

Loading differs by backend:

- CUDA/ROCm: `device_map: auto` may be used when Accelerate supports the
  installed stack.
- MPS: prefer explicit single-device placement for a model that fits; use
  explicit CPU/offload policy rather than assuming `device_map: auto` handles
  unified memory correctly.
- CPU: explicit CPU placement.

Hooks always use the device of the hidden tensor they receive. J-Lens matrices
and outputs are transferred deliberately and never through a global
`input_device` assumption.

## Optional fallback behavior

MPS CPU fallback for unsupported operations must be opt-in and recorded.
Environment variables such as `PYTORCH_ENABLE_MPS_FALLBACK=1` need to be set
before process startup, so the CLI should diagnose and explain them rather than
trying to set them after importing PyTorch.

Two runtime modes:

- `strict`: any unsupported operation or unexpected CPU fallback fails;
- `compatible`: approved operations may run on CPU and are listed in metadata.

Scientific baseline/intervention comparisons must use the same fallback mode.

## Platform-specific dependencies

Use uv with one common project definition, committed lock data, explicit
PyTorch sources, and isolated virtual environments:

```text
.venv-mps
.venv-rocm
.venv-cuda
.venv-cpu
```

The host launcher sets `UV_PROJECT_ENVIRONMENT` and invokes uv's selected
PyTorch backend. CUDA and ROCm indexes must be explicit so they cannot provide
unrelated packages. Experiment hosts use `uv sync --locked`, and each run
records the uv version, lock checksum, selected environment, and resolved torch
build.

An official ROCm container is the preferred reproducible Radeon environment.
macOS uses a native arm64 environment because Linux containers cannot expose
MPS.

The complete uv and workload-tier design is in
[`08-uv-environments-and-run-tiers.md`](08-uv-environments-and-run-tiers.md).

## Environment metadata

`environment.json` must record:

- OS name/version and architecture;
- Python and package versions;
- PyTorch version and build identifiers;
- resolved backend (`mps`, `cuda`, `rocm`, or `cpu`);
- device names/count and total/available memory where exposed;
- CUDA version or HIP/ROCm version;
- macOS and Metal/MPS availability;
- requested and resolved dtype/device map;
- offload and fallback settings;
- capability-probe results;
- deterministic-algorithm settings.

Comparison reports warn when environments differ. Cross-hardware results may be
reported as replications, but the primary intervention comparison should use
the same hardware/backend because greedy floating-point generation can still
diverge across devices.

## Portability test matrix

### Required on every pull request

- Linux CPU: unit and tiny-model integration tests.
- macOS arm64/MPS: device/dtype resolver and tiny-model baseline/capture/no-op.

### Required before a release

- NVIDIA CUDA: tiny-model and five-example Qwen-compatible smoke.
- Radeon 8060S/ROCm: same smoke suite in the documented official environment.
- CPU: evaluator/report regeneration from saved fixtures.

### Required per target before a full experiment

1. platform diagnostics;
2. operation/dtype capability probes;
3. memory preflight;
4. tiny-model capture and no-op;
5. one-example real-model baseline;
6. five-example baseline/no-op equality;
7. memory-stability check across repeated examples.

## Platform acceptance criteria

A backend is marked validated only when:

- installation steps are documented and reproducible;
- the model and fitted lens pass compatibility checks;
- all required operations pass the capability probe;
- observation capture matches capture-disabled generation;
- no-op exactly matches baseline on that backend;
- memory remains stable across examples;
- `environment.json` accurately identifies the backend;
- a five-example run can resume and reevaluate offline.

Failures on one backend must not lead to backend-specific experiment logic.
Fixes belong in the device/linear-algebra adapters while the shared benchmark
and intervention semantics remain unchanged.
