#!/usr/bin/env bash
# One-time machine setup: download pinned repos and install Python dependencies.
# Mirrors notebook section 1: "Install dependencies and verify pinned checkouts".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
jspace_bootstrap "$@"

jspace_log "=== Step 1/3: Download pinned repositories ==="
jspace_ensure_checkouts

jspace_log "=== Step 2/3: Install Python dependencies ==="
jspace_resolve_python

jspace_log "=== Step 3/3: Verify CLI installation ==="
jspace_run_module jspace_research.phase1.cli --help >/dev/null

jspace_print_runtime_info
jspace_log "Setup complete."
