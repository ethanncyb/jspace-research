#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
jspace_bootstrap
JSPACE_PHASE1_SELECTION="${JSPACE_PHASE1_DIR}/selected_layers.json"
if [[ ! -f "${JSPACE_PHASE1_SELECTION}" ]]; then
  JSPACE_PHASE1_SELECTION="${JSPACE_PHASE1_DIR}/selected_layer.json"
fi
PHASE3_K_ARGS=()
if [[ -n "${JSPACE_K:-}" ]]; then
  PHASE3_K_ARGS=(--k "${JSPACE_K}")
fi
jspace_run_module jspace_research.phase3.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --phase1 "${JSPACE_PHASE1_SELECTION}" \
  --output-dir "${JSPACE_PHASE3_DIR}" \
  "${PHASE3_K_ARGS[@]}"
