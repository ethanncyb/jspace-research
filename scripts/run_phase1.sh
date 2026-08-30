#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
jspace_bootstrap "$@"

STAGE="${STAGE:-all}"

jspace_resolve_python
jspace_validate_config
jspace_print_run_paths
jspace_phase1_extra_args

jspace_run_module jspace_research.phase1.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --output-dir "${JSPACE_PHASE1_DIR}" \
  --stage "${STAGE}" \
  "${JSPACE_PHASE1_EXTRA_ARGS[@]}"
