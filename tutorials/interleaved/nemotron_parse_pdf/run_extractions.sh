#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/direct_launcher.sh"

ROOT=${ROOT:?set ROOT to the DC10K experiment directory}
RUN_ROOT=${ROOT}/extraction
SERVER_LOG_DIR=${RUN_ROOT}/server-logs
prepare_direct_launcher

case "${1:-both}" in
  both) labels=(capped50 uncapped) ;;
  capped50|uncapped) labels=("$1") ;;
  *) echo "Usage: $0 [both|capped50|uncapped]" >&2; exit 2 ;;
esac
declare -A manifests=(
  [capped50]="${ROOT}/manifest-capped50.jsonl"
  [uncapped]="${ROOT}/manifest-uncapped-window50.jsonl"
)
declare -A max_pages=([capped50]=50 [uncapped]=1000000)
for label in "${labels[@]}"; do
  if [[ ! -f "${manifests[${label}]}" ]]; then
    echo "Missing manifest: ${manifests[${label}]}" >&2
    exit 1
  fi
  [[ ! -e "${RUN_ROOT}/${label}" ]]
done

start_direct_servers
for label in "${labels[@]}"; do
  run_dir=${RUN_ROOT}/${label}
  if [[ -e "${run_dir}/run-summary.json" || -d "${run_dir}/output" ]]; then
    echo "Refusing to overwrite existing run: ${run_dir}" >&2
    exit 1
  fi
  record_direct_benchmark "${run_dir}" "${manifests[${label}]}" "dc10k-${label}" "${max_pages[${label}]}"
done
