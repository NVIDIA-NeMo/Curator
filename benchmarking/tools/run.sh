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

# Assume this script is in the <repo_root>/benchmarking/tools directory
THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-""}
SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:-""}
GDRIVE_FOLDER_ID=${GDRIVE_FOLDER_ID:-""}
GDRIVE_SERVICE_ACCOUNT_FILE=${GDRIVE_SERVICE_ACCOUNT_FILE:-""}
NVIDIA_API_KEY=${NVIDIA_API_KEY:-""}

# get the following vars from the command line, config file(s), etc. and
# set them in this environment:
#   BASH_ENTRYPOINT_OVERRIDE
#   CURATOR_BENCHMARKING_IMAGE
#   GPUS
#   CONTAINER_MEMORY_BYTES
#   SHM_SIZE_BYTES
#   HOST_CURATOR_DIR
#   CURATOR_BENCHMARKING_DEBUG
#   VOLUME_MOUNTS
#   ENTRYPOINT_ARGS
eval_str=$(python ${THIS_SCRIPT_DIR}/gen_runscript_vars.py "${BASH_SOURCE[0]}" "$@")
eval "$eval_str"

# Get the image digest/ID for benchmark reports. This is not known at image build time.
IMAGE_DIGEST=$(docker image inspect ${CURATOR_BENCHMARKING_IMAGE} --format '{{.Digest}}' 2>/dev/null) || true
if [ -z "${IMAGE_DIGEST}" ] || [ "${IMAGE_DIGEST}" = "<none>" ]; then
    # Use the image ID as a fallback
    IMAGE_DIGEST=$(docker image inspect ${CURATOR_BENCHMARKING_IMAGE} --format '{{.ID}}' 2>/dev/null) || true
fi
if [ -z "${IMAGE_DIGEST}" ] || [ "${IMAGE_DIGEST}" = "<none>" ]; then
    IMAGE_DIGEST="<unknown>"
fi

################################################################################################################
docker run \
  --rm \
  --net=host \
  --interactive \
  --tty \
  \
  --gpus="\"${GPUS}\"" \
  --memory=${CONTAINER_MEMORY_BYTES} \
  --shm-size=${SHM_SIZE_BYTES} \
  \
  ${VOLUME_MOUNTS} \
  \
  --env=IMAGE_DIGEST=${IMAGE_DIGEST} \
  --env=MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
  --env=SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL} \
  --env=GDRIVE_FOLDER_ID=${GDRIVE_FOLDER_ID} \
  --env=GDRIVE_SERVICE_ACCOUNT_FILE=${GDRIVE_SERVICE_ACCOUNT_FILE} \
  --env=CURATOR_BENCHMARKING_DEBUG=${CURATOR_BENCHMARKING_DEBUG} \
  --env=HOST_HOSTNAME=$(hostname) \
  --env=NVIDIA_API_KEY=${NVIDIA_API_KEY} \
  \
  ${BASH_ENTRYPOINT_OVERRIDE} \
  ${CURATOR_BENCHMARKING_IMAGE} \
    "${ENTRYPOINT_ARGS[@]}"

exit $?
