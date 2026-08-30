#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
jspace_bootstrap "$@"

SKIP_SETUP="${SKIP_SETUP:-0}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"

if [[ "${SKIP_SETUP}" != "1" ]]; then
  "${SCRIPT_DIR}/setup.sh"
fi

jspace_resolve_python
jspace_validate_config
jspace_check_credentials
jspace_print_run_paths
jspace_phase1_extra_args

if [[ "${SKIP_GPU_CHECK}" != "1" ]]; then
  jspace_log "All visible GPUs on this machine:"
  jspace_list_physical_gpus
  echo
  jspace_validate_gpu_index
  echo
fi

jspace_run_module jspace_research.phase1.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --output-dir "${JSPACE_PHASE1_DIR}" \
  --stage all \
  "${JSPACE_PHASE1_EXTRA_ARGS[@]}"

jspace_run_module jspace_research.phase2.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --phase1 "${JSPACE_PHASE1_DIR}/selected_layer.json" \
  --output-dir "${JSPACE_PHASE2_DIR}" \
  --stage generate

jspace_run_module jspace_research.phase2.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --phase1 "${JSPACE_PHASE1_DIR}/selected_layer.json" \
  --output-dir "${JSPACE_PHASE2_DIR}" \
  --stage analyze

jspace_run_module jspace_research.phase3.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --phase1 "${JSPACE_PHASE1_DIR}/selected_layer.json" \
  --output-dir "${JSPACE_PHASE3_DIR}"

jspace_run_module jspace_research.phase4.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --phase1 "${JSPACE_PHASE1_DIR}/selected_layer.json" \
  --phase3 "${JSPACE_PHASE3_DIR}" \
  --bipia-root "${JSPACE_BIPIA_ROOT}" \
  --agentdojo-root "${JSPACE_AGENTDOJO_CHECKOUT}" \
  --injecagent-root "${JSPACE_INJECAGENT_CHECKOUT}" \
  --output-dir "${JSPACE_PHASE4_DIR}" \
  --stage generate

jspace_run_module jspace_research.phase4.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --phase1 "${JSPACE_PHASE1_DIR}/selected_layer.json" \
  --phase3 "${JSPACE_PHASE3_DIR}" \
  --bipia-root "${JSPACE_BIPIA_ROOT}" \
  --agentdojo-root "${JSPACE_AGENTDOJO_CHECKOUT}" \
  --injecagent-root "${JSPACE_INJECAGENT_CHECKOUT}" \
  --output-dir "${JSPACE_PHASE4_DIR}" \
  --stage analyze

jspace_assert_run_complete
