#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
jspace_bootstrap "$@"

jspace_resolve_python
jspace_validate_config
jspace_print_run_paths

jspace_run_module jspace_research.phase3.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --phase1 "${JSPACE_PHASE1_DIR}/selected_layer.json" \
  --output-dir "${JSPACE_PHASE3_DIR}"
