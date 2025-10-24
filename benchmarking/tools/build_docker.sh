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
