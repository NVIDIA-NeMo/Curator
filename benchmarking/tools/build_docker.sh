#!/bin/bash

# Exit immediately on error, unset vars are errors, pipeline errors are errors
set -euo pipefail

# Assume this script is in the <repo_root>benchmarking/tools directory
THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_DIR="$(cd ${THIS_SCRIPT_DIR}/../.. && pwd)"

# Build the base image
#docker build -f ${CURATOR_DIR}/benchmarking/Dockerfile --target curator_system_base --tag=curator_system_base ${CURATOR_DIR}

# Build the deps image
#docker build -f ${CURATOR_DIR}/benchmarking/Dockerfile --target curator_setup_deps --tag=curator_setup_deps ${CURATOR_DIR}

# Build the benchmarking image
docker build -f ${CURATOR_DIR}/benchmarking/Dockerfile --target curator_benchmarking --tag=curator_benchmarking ${CURATOR_DIR}

