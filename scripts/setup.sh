#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
command -v uv >/dev/null 2>&1 || jspace_die "uv is required: https://docs.astral.sh/uv/"
( cd "${JSPACE_REPO_ROOT}" && uv sync --extra phase4 )
jspace_ensure_bipia
mkdir -p "${JSPACE_BENCHMARKS_ROOT}"
jspace_ensure_checkout "AgentDojo" "https://github.com/ethz-spylab/agentdojo.git" \
  "${AGENTDOJO_REVISION}" "${JSPACE_AGENTDOJO_CHECKOUT}"
jspace_ensure_checkout "InjecAgent" "https://github.com/uiuc-kang-lab/InjecAgent.git" \
  "${INJECAGENT_REVISION}" "${JSPACE_INJECAGENT_CHECKOUT}"
jspace_log "Research revision: $(git -C "${JSPACE_REPO_ROOT}" rev-parse HEAD)"
jspace_run_module jspace_research.phase1.cli --help >/dev/null
jspace_log "Setup complete."
