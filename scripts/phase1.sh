#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
STAGE="${STAGE:-all}"
jspace_bootstrap
jspace_check_gpu
jspace_run_module jspace_research.phase1.cli \
  --config "${JSPACE_CONFIG_PATH}" \
  --bipia-root "${JSPACE_BIPIA_ROOT}" \
  --output-dir "${JSPACE_PHASE1_DIR}" \
  --stage "${STAGE}" \
  "${JSPACE_PHASE1_EXTRA_ARGS[@]}"
