#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
STAGE="${STAGE:-all}"
jspace_bootstrap
PHASE2_BASE=(
  --config "${JSPACE_CONFIG_PATH}"
  --phase1 "${JSPACE_PHASE1_DIR}/selected_layer.json"
  --output-dir "${JSPACE_PHASE2_DIR}"
)
case "${STAGE}" in
  generate)
    jspace_check_gpu
    jspace_run_module jspace_research.phase2.cli "${PHASE2_BASE[@]}" --stage generate ;;
  analyze)
    jspace_check_openrouter
    jspace_run_module jspace_research.phase2.cli "${PHASE2_BASE[@]}" --stage analyze ;;
  all)
    jspace_check_gpu
    jspace_run_module jspace_research.phase2.cli "${PHASE2_BASE[@]}" --stage generate
    jspace_check_openrouter
    jspace_run_module jspace_research.phase2.cli "${PHASE2_BASE[@]}" --stage analyze ;;
  *) jspace_die "unknown STAGE=${STAGE} (expected generate, analyze, or all)" ;;
esac
