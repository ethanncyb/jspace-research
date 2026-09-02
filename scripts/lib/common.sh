#!/usr/bin/env bash
# Shared helpers for phase launchers. Source from scripts/; do not execute directly.

if [[ -n "${JSPACE_COMMON_SOURCED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
JSPACE_COMMON_SOURCED=1

set -euo pipefail

BIPIA_REVISION='a004b69ec0dd446e0afd461d98cb5e96e120a5d0'
AGENTDOJO_REVISION='089ed468cf3ed0322acc66b0211f26d9d90dbf60'
INJECAGENT_REVISION='f19c9f2c79a41046eb13c03c51a24c567a8ffa07'

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

JSPACE_CONFIG_PATH="${JSPACE_CONFIG_PATH:-${JSPACE_REPO_ROOT}/configs/phase1_qwen35_9b_smoke.yaml}"
JSPACE_PHYSICAL_GPU_INDEX="${JSPACE_PHYSICAL_GPU_INDEX:-0}"
JSPACE_BENCHMARKS_ROOT="${JSPACE_BENCHMARKS_ROOT:-${JSPACE_REPO_ROOT}/../jspace-benchmarks}"
JSPACE_BIPIA_ROOT="${JSPACE_BIPIA_ROOT:-${JSPACE_REPO_ROOT}/BIPIA/benchmark}"
JSPACE_AGENTDOJO_CHECKOUT="${JSPACE_AGENTDOJO_CHECKOUT:-${JSPACE_BENCHMARKS_ROOT}/agentdojo}"
JSPACE_INJECAGENT_CHECKOUT="${JSPACE_INJECAGENT_CHECKOUT:-${JSPACE_BENCHMARKS_ROOT}/InjecAgent}"

if [[ -z "${JSPACE_RUN_ROOT:-}" ]]; then
  JSPACE_RUN_ROOT="${JSPACE_REPO_ROOT}/artifacts/jspace-qwen35_9b-smoke-gpu${JSPACE_PHYSICAL_GPU_INDEX}"
fi

JSPACE_PHASE1_DIR="${JSPACE_PHASE1_DIR:-${JSPACE_RUN_ROOT}/phase1}"
JSPACE_PHASE2_DIR="${JSPACE_PHASE2_DIR:-${JSPACE_RUN_ROOT}/phase2}"
JSPACE_PHASE3_DIR="${JSPACE_PHASE3_DIR:-${JSPACE_RUN_ROOT}/phase3}"
JSPACE_PHASE4_DIR="${JSPACE_PHASE4_DIR:-${JSPACE_RUN_ROOT}/phase4}"

JSPACE_PHASE1_EXTRA_ARGS=()

jspace_log() { printf '%s\n' "$*"; }
jspace_die() { echo "error: $*" >&2; exit 1; }

jspace_command_env() { export CUDA_VISIBLE_DEVICES="${JSPACE_PHYSICAL_GPU_INDEX}"; }

jspace_run_module() {
  local module="$1"
  shift
  jspace_command_env
  ( cd "${JSPACE_REPO_ROOT}" && uv run python -m "${module}" "$@" )
}

jspace_validate_config() {
  [[ -f "${JSPACE_CONFIG_PATH}" ]] || jspace_die "config not found: ${JSPACE_CONFIG_PATH}"
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

jspace_check_gpu() {
  jspace_command_env
  ( cd "${JSPACE_REPO_ROOT}" && uv run python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA is not available.", file=sys.stderr)
    raise SystemExit(1)
print("GPU:", torch.cuda.get_device_name(0))
PY
  ) || jspace_die "CUDA GPU required for this stage"
}

jspace_check_openrouter() {
  [[ -n "${OPENROUTER_API_KEY:-}" ]] || jspace_die "OPENROUTER_API_KEY is required for semantic BIPIA judging"
}

jspace_print_run_paths() {
  jspace_log "Config: ${JSPACE_CONFIG_PATH}"
  jspace_log "Run root: ${JSPACE_RUN_ROOT}"
  jspace_log "BIPIA root: ${JSPACE_BIPIA_ROOT}"
  jspace_log "Phase 1: ${JSPACE_PHASE1_DIR}"
  jspace_log "Phase 2: ${JSPACE_PHASE2_DIR}"
  jspace_log "Phase 3: ${JSPACE_PHASE3_DIR}"
  jspace_log "Phase 4: ${JSPACE_PHASE4_DIR}"
  jspace_log "Physical GPU index: ${JSPACE_PHYSICAL_GPU_INDEX}"
}

jspace_bootstrap() {
  jspace_validate_config
  jspace_phase1_extra_args
  jspace_print_run_paths
  mkdir -p "${JSPACE_PHASE1_DIR}" "${JSPACE_PHASE2_DIR}" "${JSPACE_PHASE3_DIR}" "${JSPACE_PHASE4_DIR}"
}

jspace_ensure_checkout() {
  local name="$1" url="$2" revision="$3" dest="$4"
  if [[ ! -d "${dest}/.git" ]]; then
    jspace_log "Cloning ${name} to ${dest}"
    git clone "${url}" "${dest}"
  fi
  git -C "${dest}" fetch --all --tags >/dev/null 2>&1 || true
  git -C "${dest}" checkout "${revision}"
  jspace_log "${name} revision: $(git -C "${dest}" rev-parse HEAD)"
}

jspace_ensure_bipia() {
  local checkout="${JSPACE_REPO_ROOT}/BIPIA"
  if [[ -f "${JSPACE_REPO_ROOT}/.gitmodules" ]]; then
    git -C "${JSPACE_REPO_ROOT}" submodule update --init BIPIA
  elif [[ ! -d "${checkout}/.git" ]]; then
    jspace_ensure_checkout "BIPIA" "https://github.com/microsoft/BIPIA.git" \
      "${BIPIA_REVISION}" "${checkout}"
  else
    git -C "${checkout}" checkout "${BIPIA_REVISION}"
  fi
  jspace_log "BIPIA revision: $(git -C "${checkout}" rev-parse HEAD)"
}
