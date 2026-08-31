#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
jspace_bootstrap "$@"

STAGE="${STAGE:-generate}"

jspace_resolve_python
jspace_validate_config
jspace_print_run_paths

if [[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]]; then
  jspace_check_credentials
fi

jspace_run_module jspace_research.phase2.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --phase1 "${JSPACE_PHASE1_DIR}/selected_layer.json" \
  --output-dir "${JSPACE_PHASE2_DIR}" \
  --stage "${STAGE}"
