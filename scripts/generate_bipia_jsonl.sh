#!/usr/bin/env bash
# Generate BIPIA WebQA and Summarization train.jsonl / test.jsonl for full runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

TASKS="all"
NEWSQA_DIR=""

usage() {
  cat <<'EOF'
Usage: ./scripts/generate_bipia_jsonl.sh [options]

Generate licensed BIPIA context files required for full Phase 1 / Phase 4 runs.

Options:
  --tasks TASKS         webqa | summarization | all (default: all)
  --newsqa-dir PATH     Directory with NewsQA combined-csv files (required for webqa)
  --config PATH         Launcher config (default: scripts/config.yaml)
  -h, --help            Show this help

WebQA prerequisites (see BIPIA/benchmark/README.md):
  Download NewsQA and place these files in --newsqa-dir:
    combined-newsqa-data-v1.csv
    combined-newsqa-data-v1.json

Summarization prerequisites:
  Accept the XSum license on Hugging Face (EdinburghNLP/xsum).
  The script downloads XSum via the datasets library.

Outputs (written in place under BIPIA/benchmark/):
  qa/train.jsonl, qa/test.jsonl
  abstract/train.jsonl, abstract/test.jsonl

After generation, point scripts/config.yaml at the train files:
  paths:
    webqa_train_path: BIPIA/benchmark/qa/train.jsonl
    summarization_train_path: BIPIA/benchmark/abstract/train.jsonl
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      [[ $# -ge 2 ]] || jspace_die "--tasks requires a value"
      TASKS="$2"
      shift 2
      ;;
    --newsqa-dir)
      [[ $# -ge 2 ]] || jspace_die "--newsqa-dir requires a path"
      NEWSQA_DIR="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || jspace_die "--config requires a path"
      JSPACE_LAUNCHER_CONFIG="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      jspace_die "unknown argument: $1 (try --help)"
      ;;
  esac
done

jspace_bootstrap --config "${JSPACE_LAUNCHER_CONFIG:-${JSPACE_REPO_ROOT}/scripts/config.yaml}"
jspace_resolve_python

QA_DIR="${JSPACE_BIPIA_ROOT}/benchmark/qa"
ABSTRACT_DIR="${JSPACE_BIPIA_ROOT}/benchmark/abstract"

jspace_ensure_bipia_data_deps() {
  if ! "${JSPACE_PYTHON}" -c 'import datasets, jsonlines' >/dev/null 2>&1; then
    jspace_log "Installing datasets for BIPIA context generation"
    if [[ "${JSPACE_RUNTIME_MODE:-}" == *"system python"* ]]; then
      "${JSPACE_PYTHON}" -Im pip install --user 'datasets>=2.8.0' jsonlines
    else
      "${JSPACE_PYTHON}" -Im pip install 'datasets>=2.8.0' jsonlines
    fi
  fi
  "${JSPACE_PYTHON}" -c 'import datasets, jsonlines'
}

jspace_generate_webqa() {
  [[ -n "${NEWSQA_DIR}" ]] || jspace_die "--newsqa-dir is required for WebQA generation"
  local newsqa_path
  newsqa_path="$(cd "${NEWSQA_DIR}" && pwd)"
  [[ -f "${newsqa_path}/combined-newsqa-data-v1.csv" ]] \
    || jspace_die "missing ${newsqa_path}/combined-newsqa-data-v1.csv"
  [[ -f "${newsqa_path}/combined-newsqa-data-v1.json" ]] \
    || jspace_die "missing ${newsqa_path}/combined-newsqa-data-v1.json"
  jspace_log "Generating WebQA train.jsonl and test.jsonl in ${QA_DIR}"
  (
    cd "${QA_DIR}"
    "${JSPACE_PYTHON}" process.py --data_dir "${newsqa_path}"
  )
}

jspace_generate_summarization() {
  jspace_log "Generating Summarization train.jsonl and test.jsonl in ${ABSTRACT_DIR}"
  jspace_log "You must have accepted the XSum dataset license on Hugging Face."
  (
    cd "${ABSTRACT_DIR}"
    printf 'y\n' | "${JSPACE_PYTHON}" process.py
  )
}

jspace_ensure_bipia_data_deps

case "${TASKS}" in
  webqa)
    jspace_generate_webqa
    ;;
  summarization)
    jspace_generate_summarization
    ;;
  all)
    jspace_generate_webqa
    jspace_generate_summarization
    ;;
  *)
    jspace_die "unknown --tasks value: ${TASKS} (expected webqa, summarization, or all)"
    ;;
esac

jspace_log "Generation complete."
jspace_log "WebQA train: ${QA_DIR}/train.jsonl"
jspace_log "WebQA test: ${QA_DIR}/test.jsonl"
jspace_log "Summarization train: ${ABSTRACT_DIR}/train.jsonl"
jspace_log "Summarization test: ${ABSTRACT_DIR}/test.jsonl"
jspace_log "Set in scripts/config.yaml:"
jspace_log "  paths.webqa_train_path: BIPIA/benchmark/qa/train.jsonl"
jspace_log "  paths.summarization_train_path: BIPIA/benchmark/abstract/train.jsonl"
