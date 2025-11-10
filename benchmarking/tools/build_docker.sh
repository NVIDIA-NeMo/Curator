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

# Make the image tags unique based on the current timestamp
# Use UTC to avoid confusion between users/build machines in different timezones
UTC_TIMESTAMP=$(date --utc "+%Y%m%d%H%M%SUTC")
NEMO_CURATOR_TAG="nemo_curator:${UTC_TIMESTAMP}"
NEMO_CURATOR_BENCHMARKING_TAG="nemo_curator_benchmarking:${UTC_TIMESTAMP}"

# Assume this script is in the <repo_root>benchmarking/tools directory
THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_DIR="$(cd ${THIS_SCRIPT_DIR}/../.. && pwd)"

# Build the standard NeMo Curator image
docker build \
  -f ${CURATOR_DIR}/docker/Dockerfile \
  --target nemo_curator \
  --tag=${NEMO_CURATOR_TAG} \
  --tag=nemo_curator:latest \
  ${CURATOR_DIR}

# Build the benchmarking image which extends the standard NeMo Curator image
docker build \
  -f ${CURATOR_DIR}/benchmarking/Dockerfile \
  --target nemo_curator_benchmarking \
  --tag=${NEMO_CURATOR_BENCHMARKING_TAG} \
  --tag=nemo_curator_benchmarking:latest \
  --build-arg NEMO_CURATOR_IMAGE=${NEMO_CURATOR_TAG} \
  ${CURATOR_DIR}
