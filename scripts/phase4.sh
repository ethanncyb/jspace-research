#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
jspace_bootstrap "$@"

STAGE="${STAGE:-all}"

jspace_resolve_python
jspace_validate_config
jspace_check_phase4_dependencies
jspace_print_run_paths

if [[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]]; then
  jspace_check_credentials
fi

jspace_run_module jspace_research.phase4.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --phase1 "${JSPACE_PHASE1_DIR}/selected_layer.json" \
  --phase3 "${JSPACE_PHASE3_DIR}" \
  --bipia-root "${JSPACE_BIPIA_ROOT}" \
  --agentdojo-root "${JSPACE_AGENTDOJO_CHECKOUT}" \
  --injecagent-root "${JSPACE_INJECAGENT_CHECKOUT}" \
  --output-dir "${JSPACE_PHASE4_DIR}" \
  --stage "${STAGE}"
