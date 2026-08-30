#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
jspace_bootstrap "$@"

jspace_resolve_python
jspace_ensure_checkouts
jspace_print_runtime_info

jspace_run_module jspace_research.phase1.cli --help >/dev/null
jspace_log "Setup complete."
