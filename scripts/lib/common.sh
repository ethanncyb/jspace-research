#!/usr/bin/env bash
# Shared helpers for terminal launchers. Source from scripts/; do not execute directly.

if [[ -n "${JSPACE_COMMON_SOURCED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
JSPACE_COMMON_SOURCED=1

set -euo pipefail

AGENTDOJO_REVISION='089ed468cf3ed0322acc66b0211f26d9d90dbf60'
INJECAGENT_REVISION='f19c9f2c79a41046eb13c03c51a24c567a8ffa07'
BIPIA_REVISION='a004b69ec0dd446e0afd461d98cb5e96e120a5d0'
PIP_EXTRAS='phase4'

jspace_repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
  if [[ -f "${here}/pyproject.toml" ]]; then
    printf '%s\n' "${here}"
    return 0
  fi
  here="$(cd "${here}/.." && pwd)"
  if [[ -f "${here}/pyproject.toml" ]]; then
    printf '%s\n' "${here}"
    return 0
  fi
  echo "Could not locate repository root (expected pyproject.toml)." >&2
  return 1
}

JSPACE_REPO_ROOT="$(jspace_repo_root)"

jspace_log() {
  printf '%s\n' "$*"
}

jspace_die() {
  echo "error: $*" >&2
  exit 1
}

jspace_parse_launcher_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        [[ $# -ge 2 ]] || jspace_die "--config requires a path"
        JSPACE_LAUNCHER_CONFIG="$2"
        shift 2
        ;;
      --local-config)
        [[ $# -ge 2 ]] || jspace_die "--local-config requires a path"
        JSPACE_LOCAL_CONFIG="$2"
        shift 2
        ;;
      --run-config)
        [[ $# -ge 2 ]] || jspace_die "--run-config requires a path"
        JSPACE_RUN_CONFIG="$2"
        shift 2
        ;;
      --env-config)
        [[ $# -ge 2 ]] || jspace_die "--env-config requires a path"
        JSPACE_ENV_CONFIG="$2"
        shift 2
        ;;
      -h | --help)
        cat <<'EOF'
Usage: ./scripts/<script>.sh [--config PATH] [--local-config PATH]

  --config PATH         Unified launcher YAML (default: scripts/config.yaml)
  --local-config PATH   Optional local overrides (default: scripts/config.local.yaml)

Legacy (still supported):
  --run-config PATH     Old launcher run YAML
  --env-config PATH     Old launcher env YAML

Environment variables override YAML values.
EOF
        exit 0
        ;;
      *)
        jspace_die "unknown argument: $1 (try --help)"
        ;;
    esac
  done
}

jspace_apply_launcher_configs() {
  local loader="${JSPACE_REPO_ROOT}/scripts/lib/load_launcher_config.py"
  local default_config="${JSPACE_REPO_ROOT}/scripts/config.yaml"
  local launcher_config="${JSPACE_LAUNCHER_CONFIG:-}"
  local local_config="${JSPACE_LOCAL_CONFIG:-}"
  local run_config="${JSPACE_RUN_CONFIG:-}"
  local env_config="${JSPACE_ENV_CONFIG:-}"
  local args=(--repo-root "${JSPACE_REPO_ROOT}")
  local has_config=0

  if [[ -z "${launcher_config}" && -f "${default_config}" ]]; then
    launcher_config="${default_config}"
  fi

  if [[ -n "${launcher_config}" && -f "${launcher_config}" ]]; then
    args+=(--config "${launcher_config}")
    has_config=1
  fi
  if [[ -n "${local_config}" && -f "${local_config}" ]]; then
    args+=(--local-config "${local_config}")
    has_config=1
  fi
  if [[ -n "${run_config}" && -f "${run_config}" ]]; then
    args+=(--run-config "${run_config}")
    has_config=1
  fi
  if [[ -n "${env_config}" && -f "${env_config}" ]]; then
    args+=(--env-config "${env_config}")
    has_config=1
  fi

  if [[ "${has_config}" -eq 0 ]]; then
    return 0
  fi

  local loader_python="python3"
  if ! python3 -c 'import yaml' >/dev/null 2>&1; then
    if [[ -x "${JSPACE_VENV_DIR}/bin/python" ]] \
      && "${JSPACE_VENV_DIR}/bin/python" -c 'import yaml' >/dev/null 2>&1; then
      loader_python="${JSPACE_VENV_DIR}/bin/python"
    else
      jspace_log "PyYAML is not installed; skipping launcher YAML. Run ./scripts/setup.sh or export JSPACE_* vars."
      return 0
    fi
  fi

  jspace_log "Loading launcher config"
  [[ -n "${launcher_config}" && -f "${launcher_config}" ]] && jspace_log "  config: ${launcher_config}"
  [[ -n "${local_config}" && -f "${local_config}" ]] && jspace_log "  local: ${local_config}"
  [[ -n "${run_config}" && -f "${run_config}" ]] && jspace_log "  run: ${run_config}"
  [[ -n "${env_config}" && -f "${env_config}" ]] && jspace_log "  env: ${env_config}"
  eval "$("${loader_python}" "${loader}" "${args[@]}")"
}

jspace_init_run_paths() {
  JSPACE_BENCHMARKS_ROOT="${JSPACE_BENCHMARKS_ROOT:-${JSPACE_REPO_ROOT}/../jspace-benchmarks}"
  JSPACE_USE_PROJECT_VENV="${JSPACE_USE_PROJECT_VENV:-1}"
  JSPACE_VENV_DIR="${JSPACE_VENV_DIR:-${JSPACE_REPO_ROOT}/.venv}"
  JSPACE_MODEL_KEY="${JSPACE_MODEL_KEY:-qwen35_9b}"
  JSPACE_RUN_MODE="${JSPACE_RUN_MODE:-smoke}"
  JSPACE_PHYSICAL_GPU_INDEX="${JSPACE_PHYSICAL_GPU_INDEX:-0}"

  JSPACE_CONFIG_NAME="phase1_${JSPACE_MODEL_KEY}_${JSPACE_RUN_MODE}"
  JSPACE_RUN_NAME="jspace-${JSPACE_MODEL_KEY}-${JSPACE_RUN_MODE}-gpu${JSPACE_PHYSICAL_GPU_INDEX}"
  JSPACE_CONFIG_PATH="${JSPACE_CONFIG_PATH:-${JSPACE_REPO_ROOT}/configs/${JSPACE_CONFIG_NAME}.yaml}"
  JSPACE_RUN_ROOT="${JSPACE_RUN_ROOT:-${JSPACE_REPO_ROOT}/artifacts/${JSPACE_RUN_NAME}}"

  JSPACE_BIPIA_CHECKOUT="${JSPACE_BIPIA_CHECKOUT:-${JSPACE_REPO_ROOT}/BIPIA}"
  JSPACE_AGENTDOJO_CHECKOUT="${JSPACE_AGENTDOJO_CHECKOUT:-${JSPACE_BENCHMARKS_ROOT}/agentdojo}"
  JSPACE_INJECAGENT_CHECKOUT="${JSPACE_INJECAGENT_CHECKOUT:-${JSPACE_BENCHMARKS_ROOT}/InjecAgent}"
  JSPACE_BIPIA_ROOT="${JSPACE_BIPIA_ROOT:-${JSPACE_BIPIA_CHECKOUT}/benchmark}"

  JSPACE_PHASE1_DIR="${JSPACE_PHASE1_DIR:-${JSPACE_RUN_ROOT}/phase1}"
  JSPACE_PHASE2_DIR="${JSPACE_PHASE2_DIR:-${JSPACE_RUN_ROOT}/phase2}"
  JSPACE_PHASE3_DIR="${JSPACE_PHASE3_DIR:-${JSPACE_RUN_ROOT}/phase3}"
  JSPACE_PHASE4_DIR="${JSPACE_PHASE4_DIR:-${JSPACE_RUN_ROOT}/phase4}"
}

jspace_bootstrap() {
  jspace_parse_launcher_args "$@"
  jspace_apply_launcher_configs
  jspace_init_run_paths
}

jspace_venv_python() {
  local bin_dir="${JSPACE_VENV_DIR}/bin"
  local candidate
  for candidate in \
    "${bin_dir}/python" \
    "${bin_dir}/python3" \
    "${bin_dir}/python$(python3 -c 'import sys; print(sys.version_info.major)')" \
    "${bin_dir}/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

jspace_venv_is_valid() {
  local venv_python
  [[ -f "${JSPACE_VENV_DIR}/pyvenv.cfg" ]] || return 1
  venv_python="$(jspace_venv_python)" || return 1
  [[ -x "${venv_python}" ]] || return 1
  local prefix
  prefix="$("${venv_python}" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
  [[ -n "${prefix}" ]] || return 1
  [[ "$(cd "${prefix}" && pwd)" == "$(cd "${JSPACE_VENV_DIR}" && pwd)" ]]
}

jspace_import_check() {
  local python_executable="$1"
  local use_user_site="${2:-0}"
  local env=()
  if [[ "${use_user_site}" == "1" ]]; then
    env=(env PIP_USER=1)
  else
    env=(env PIP_USER=0 VIRTUAL_ENV="${JSPACE_VENV_DIR}" PYTHONNOUSERSITE=1)
  fi
  "${env[@]}" "${python_executable}" -c \
    'import jspace_research, pandas, torch; from jspace_research.phase1.cli import main'
}

jspace_pip_install() {
  local python_executable="$1"
  local editable="$2"
  local use_user_site="$3"
  local cmd=(
    "${python_executable}" -Im pip install
  )
  if [[ "${use_user_site}" == "1" ]]; then
    cmd+=(--user)
  else
    cmd+=(--no-user)
  fi
  if [[ "${editable}" == "1" ]]; then
    cmd+=(-e "${JSPACE_REPO_ROOT}[${PIP_EXTRAS}]")
  else
    cmd+=("${JSPACE_REPO_ROOT}[${PIP_EXTRAS}]")
  fi
  (cd "${JSPACE_REPO_ROOT}" && "${cmd[@]}")
}

jspace_create_project_venv() {
  if [[ -d "${JSPACE_VENV_DIR}" ]]; then
    rm -rf "${JSPACE_VENV_DIR}"
  fi
  python3 -m venv --clear "${JSPACE_VENV_DIR}"
  jspace_venv_is_valid || jspace_die "created ${JSPACE_VENV_DIR}, but sys.prefix does not point there"
}

jspace_try_project_venv() {
  local venv_python
  if ! jspace_venv_is_valid; then
    jspace_log "Creating project virtualenv at ${JSPACE_VENV_DIR}"
    jspace_create_project_venv || return 1
  fi
  venv_python="$(jspace_venv_python)" || return 1
  if ! jspace_import_check "${venv_python}" 0; then
    jspace_log "Installing jspace-research into project virtualenv"
    if ! jspace_pip_install "${venv_python}" 1 0; then
      jspace_log "Editable install failed; retrying non-editable install"
      jspace_pip_install "${venv_python}" 0 0 || return 1
    fi
  fi
  jspace_import_check "${venv_python}" 0 || return 1
  JSPACE_PYTHON="${venv_python}"
  JSPACE_RUNTIME_MODE='project .venv'
}

jspace_ensure_system_python() {
  local runtime_python
  runtime_python="$(command -v python3)"
  [[ -n "${runtime_python}" ]] || jspace_die "python3 not found on PATH"
  if ! jspace_import_check "${runtime_python}" 1; then
    jspace_log "Installing jspace-research with pip --user"
    if ! jspace_pip_install "${runtime_python}" 1 1; then
      jspace_log "Editable --user install failed; retrying non-editable --user install"
      jspace_pip_install "${runtime_python}" 0 1
    fi
  fi
  jspace_import_check "${runtime_python}" 1 \
    || jspace_die "system python still cannot import jspace_research after pip install --user"
  JSPACE_PYTHON="${runtime_python}"
  JSPACE_RUNTIME_MODE='system python (--user install)'
}

jspace_resolve_python() {
  if [[ -n "${JSPACE_PYTHON:-}" ]]; then
    JSPACE_RUNTIME_MODE="${JSPACE_RUNTIME_MODE:-explicit JSPACE_PYTHON}"
    return 0
  fi
  if [[ "${JSPACE_USE_PROJECT_VENV}" == "1" ]]; then
    if jspace_try_project_venv; then
      return 0
    fi
    jspace_log "Project virtualenv setup failed; falling back to system python with pip --user"
  fi
  jspace_ensure_system_python
}

jspace_pipeline_env() {
  local gpu_index="${1:-${JSPACE_PHYSICAL_GPU_INDEX}}"
  export CUDA_VISIBLE_DEVICES="${gpu_index}"
  if [[ "${JSPACE_USE_PROJECT_VENV}" == "1" && -n "${JSPACE_VENV_DIR:-}" && -d "${JSPACE_VENV_DIR}" ]]; then
    export VIRTUAL_ENV="${JSPACE_VENV_DIR}"
    export PATH="${JSPACE_VENV_DIR}/bin:${PATH}"
    export PYTHONNOUSERSITE=1
    export PIP_USER=0
  else
    unset VIRTUAL_ENV
    export PIP_USER=1
  fi
  unset PYTHONHOME
}

jspace_run_module() {
  local module="$1"
  shift
  jspace_log "Running on physical GPU ${JSPACE_PHYSICAL_GPU_INDEX}: ${JSPACE_PYTHON} -m ${module} $*"
  (
    cd "${JSPACE_REPO_ROOT}"
    jspace_pipeline_env "${JSPACE_PHYSICAL_GPU_INDEX}"
    exec "${JSPACE_PYTHON}" -m "${module}" "$@"
  )
}

jspace_ensure_checkouts() {
  jspace_log "Initializing BIPIA submodule at ${JSPACE_BIPIA_CHECKOUT}"
  git -C "${JSPACE_REPO_ROOT}" submodule update --init BIPIA
  if [[ ! -d "${JSPACE_BIPIA_CHECKOUT}" ]] || ! git -C "${JSPACE_BIPIA_CHECKOUT}" rev-parse HEAD >/dev/null 2>&1; then
    jspace_die "BIPIA checkout missing after submodule init: ${JSPACE_BIPIA_CHECKOUT}"
  fi
  jspace_log "Checking out pinned BIPIA revision ${BIPIA_REVISION}"
  git -C "${JSPACE_BIPIA_CHECKOUT}" checkout "${BIPIA_REVISION}"

  mkdir -p "${JSPACE_BENCHMARKS_ROOT}"
  if [[ ! -d "${JSPACE_AGENTDOJO_CHECKOUT}/.git" ]]; then
    jspace_log "Cloning AgentDojo into ${JSPACE_AGENTDOJO_CHECKOUT}"
    git clone https://github.com/ethz-spylab/agentdojo.git "${JSPACE_AGENTDOJO_CHECKOUT}"
  fi
  jspace_log "Checking out pinned AgentDojo revision ${AGENTDOJO_REVISION}"
  git -C "${JSPACE_AGENTDOJO_CHECKOUT}" checkout "${AGENTDOJO_REVISION}"

  if [[ ! -d "${JSPACE_INJECAGENT_CHECKOUT}/.git" ]]; then
    jspace_log "Cloning InjecAgent into ${JSPACE_INJECAGENT_CHECKOUT}"
    git clone https://github.com/uiuc-kang-lab/InjecAgent.git "${JSPACE_INJECAGENT_CHECKOUT}"
  fi
  jspace_log "Checking out pinned InjecAgent revision ${INJECAGENT_REVISION}"
  git -C "${JSPACE_INJECAGENT_CHECKOUT}" checkout "${INJECAGENT_REVISION}"
}

jspace_print_checkout_info() {
  jspace_log "Repository root: ${JSPACE_REPO_ROOT}"
  jspace_log "Benchmarks root: ${JSPACE_BENCHMARKS_ROOT}"
  jspace_log "BIPIA checkout: ${JSPACE_BIPIA_CHECKOUT}"
  jspace_log "AgentDojo checkout: ${JSPACE_AGENTDOJO_CHECKOUT}"
  jspace_log "InjecAgent checkout: ${JSPACE_INJECAGENT_CHECKOUT}"
  jspace_log "Research revision: $(git -C "${JSPACE_REPO_ROOT}" rev-parse HEAD)"
  jspace_log "BIPIA revision: $(git -C "${JSPACE_BIPIA_CHECKOUT}" rev-parse HEAD)"
  jspace_log "AgentDojo revision: $(git -C "${JSPACE_AGENTDOJO_CHECKOUT}" rev-parse HEAD)"
  jspace_log "InjecAgent revision: $(git -C "${JSPACE_INJECAGENT_CHECKOUT}" rev-parse HEAD)"
}

jspace_print_runtime_info() {
  jspace_print_checkout_info
  jspace_log "Runtime mode: ${JSPACE_RUNTIME_MODE}"
  jspace_log "Runtime python: ${JSPACE_PYTHON}"
}

jspace_check_credentials() {
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    jspace_die "OPENROUTER_API_KEY is not set. Export it before running analysis stages."
  fi
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    jspace_log "HF token not set in environment; huggingface-cli login or export HF_TOKEN before GPU stages"
  fi
}

jspace_validate_config() {
  [[ -f "${JSPACE_CONFIG_PATH}" ]] || jspace_die "configuration not found: ${JSPACE_CONFIG_PATH}"
}

jspace_print_run_paths() {
  mkdir -p "${JSPACE_RUN_ROOT}"
  jspace_log "Config: ${JSPACE_CONFIG_PATH}"
  jspace_log "Run root: ${JSPACE_RUN_ROOT}"
  jspace_log "Physical GPU index: ${JSPACE_PHYSICAL_GPU_INDEX}"
}

jspace_list_physical_gpus() {
  jspace_pipeline_env
  "${JSPACE_PYTHON}" -c \
    'import torch; count = torch.cuda.device_count(); print(f"Visible GPU count: {count}"); [print(f"GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(count)]'
}

jspace_validate_gpu_index() {
  local gpu_index="${1:-${JSPACE_PHYSICAL_GPU_INDEX}}"
  jspace_pipeline_env "${gpu_index}"
  "${JSPACE_PYTHON}" -c \
    'import os, torch; index = int(os.environ["CUDA_VISIBLE_DEVICES"]); assert torch.cuda.is_available(), "CUDA is not available"; assert torch.cuda.device_count() == 1, "Expected one visible GPU"; print("Selected physical GPU:", index); print("Logical device:", torch.cuda.get_device_name(0))'
}

jspace_phase1_extra_args() {
  JSPACE_PHASE1_EXTRA_ARGS=()
  if [[ -n "${WEBQA_TRAIN_PATH:-}" ]]; then
    JSPACE_PHASE1_EXTRA_ARGS+=(--webqa-train "${WEBQA_TRAIN_PATH}")
  fi
  if [[ -n "${SUMMARIZATION_TRAIN_PATH:-}" ]]; then
    JSPACE_PHASE1_EXTRA_ARGS+=(--summarization-train "${SUMMARIZATION_TRAIN_PATH}")
  fi
}

jspace_assert_run_complete() {
  local selection_layer
  [[ -f "${JSPACE_PHASE1_DIR}/selected_layer.json" ]] || jspace_die "missing ${JSPACE_PHASE1_DIR}/selected_layer.json"
  [[ -f "${JSPACE_PHASE2_DIR}/phase2_results.parquet" ]] || jspace_die "missing ${JSPACE_PHASE2_DIR}/phase2_results.parquet"
  [[ -f "${JSPACE_PHASE3_DIR}/mean_detector.pt" ]] || jspace_die "missing ${JSPACE_PHASE3_DIR}/mean_detector.pt"
  [[ -f "${JSPACE_PHASE3_DIR}/logistic_detector.pt" ]] || jspace_die "missing ${JSPACE_PHASE3_DIR}/logistic_detector.pt"
  [[ -f "${JSPACE_PHASE4_DIR}/phase4_predictions.parquet" ]] || jspace_die "missing ${JSPACE_PHASE4_DIR}/phase4_predictions.parquet"
  selection_layer="$("${JSPACE_PYTHON}" -c "import json; print(json.load(open('${JSPACE_PHASE1_DIR}/selected_layer.json'))['selected_layer'])")"
  jspace_log "Complete run root: ${JSPACE_RUN_ROOT}"
  jspace_log "Phase 1 selected layer: ${selection_layer}"
  jspace_log "Phase 2 results: ${JSPACE_PHASE2_DIR}/phase2_results.parquet"
  jspace_log "Phase 3 metrics: ${JSPACE_PHASE3_DIR}/phase3_metrics.csv"
  jspace_log "Phase 4 metrics: ${JSPACE_PHASE4_DIR}/phase4_metrics.csv"
}

jspace_init_run_paths
