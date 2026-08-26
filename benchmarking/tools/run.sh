#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_DIR="$(cd "${THIS_SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN=${PYTHON_BIN:-""}
if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if [ -z "${PYTHON_BIN}" ]; then
  echo "Error: benchmarking/tools/run.sh requires Python 3.11+ on the host." >&2
  exit 1
fi

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  cat <<EOF
Usage: $0 [OPTIONS] [ARGS ...]

Compatibility wrapper around:
  python benchmarking/run.py run --image <image> ...

Use curator-benchmark directly when it is installed:
  curator-benchmark run --image <image> --config benchmarking/nightly-benchmark.yaml
  curator-benchmark shell --image <image>

By default this wrapper uses:
  ${CURATOR_BENCHMARK_IMAGE:-${CURATOR_BENCHMARKING_IMAGE:-nemo_curator:latest}}

EOF
  "${PYTHON_BIN}" "${CURATOR_DIR}/benchmarking/run.py" run --help
  exit 0
fi

ARGS=("$@")
SUBCOMMAND=run
PASSTHROUGH=()
HAS_TARGET=false

while [[ ${#ARGS[@]} -gt 0 ]]; do
  case "${ARGS[0]}" in
    --shell)
      SUBCOMMAND=shell
      ARGS=("${ARGS[@]:1}")
      ;;
    --image|--container)
      HAS_TARGET=true
      PASSTHROUGH+=("${ARGS[0]}")
      ARGS=("${ARGS[@]:1}")
      if [[ ${#ARGS[@]} -gt 0 && "${ARGS[0]}" != --* ]]; then
        PASSTHROUGH+=("${ARGS[0]}")
        ARGS=("${ARGS[@]:1}")
      fi
      ;;
    *)
      PASSTHROUGH+=("${ARGS[0]}")
      ARGS=("${ARGS[@]:1}")
      ;;
  esac
done

if ! ${HAS_TARGET}; then
  PASSTHROUGH=(--image "${CURATOR_BENCHMARK_IMAGE:-${CURATOR_BENCHMARKING_IMAGE:-nemo_curator:latest}}" "${PASSTHROUGH[@]}")
fi

exec "${PYTHON_BIN}" "${CURATOR_DIR}/benchmarking/run.py" "${SUBCOMMAND}" "${PASSTHROUGH[@]}"
