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

# Tag the images with ':latest' tag if --tag-as-latest flag is present.
# This is not the default to prevent name collisions from multiple users.
TAG_AS_LATEST=false
if [[ "${*}" == *"--tag-as-latest"* ]]; then
  TAG_AS_LATEST=true
fi

UTC_TIMESTAMP=$(date --utc "+%Y%m%d%H%M%SUTC")
CURATOR_IMAGE=${CURATOR_IMAGE:-"nemo_curator:${UTC_TIMESTAMP}"}
CURATOR_BENCHMARKING_IMAGE=${CURATOR_BENCHMARKING_IMAGE:-"nemo_curator_benchmarking:${UTC_TIMESTAMP}"}

# Assume this script is in the <repo_root>benchmarking/tools directory
THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_DIR="$(cd ${THIS_SCRIPT_DIR}/../.. && pwd)"

# Build the standard NeMo Curator image
docker build \
  -f ${CURATOR_DIR}/docker/Dockerfile \
  --target nemo_curator \
  --tag=${CURATOR_IMAGE} \
  ${CURATOR_DIR}

if ${TAG_AS_LATEST}; then
  # Tag image as <name>:latest, where <name> is the part of CURATOR_IMAGE before the colon
  docker tag "${CURATOR_IMAGE}" "${CURATOR_IMAGE%%:*}:latest"
fi

# Build the benchmarking image which extends the standard NeMo Curator image
docker build \
  -f ${CURATOR_DIR}/benchmarking/Dockerfile \
  --target nemo_curator_benchmarking \
  --tag=${CURATOR_BENCHMARKING_IMAGE} \
  --build-arg CURATOR_IMAGE=${CURATOR_IMAGE} \
  ${CURATOR_DIR}

if ${TAG_AS_LATEST}; then
  docker tag "${CURATOR_BENCHMARKING_IMAGE}" "${CURATOR_BENCHMARKING_IMAGE%%:*}:latest"
fi
