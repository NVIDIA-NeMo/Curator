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
SLACK_BOT_TOKEN=${SLACK_BOT_TOKEN:-""}
SLACK_CHANNEL_ID=${SLACK_CHANNEL_ID:-""}
GDRIVE_FOLDER_ID=${GDRIVE_FOLDER_ID:-""}
GDRIVE_SERVICE_ACCOUNT_FILE=${GDRIVE_SERVICE_ACCOUNT_FILE:-""}
NVIDIA_API_KEY=${NVIDIA_API_KEY:-""}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-""}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-""}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-""}
AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL:-""}
S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-""}
CURATOR_BENCHMARK_RAY_TMP_ROOT=${CURATOR_BENCHMARK_RAY_TMP_ROOT:-""}
TMPDIR=${TMPDIR:-""}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-""}

PYTHON_BIN=${PYTHON_BIN:-""}
if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -c 'import sys; raise SystemExit(sys.version_info < (3, 9))'; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if [ -z "${PYTHON_BIN}" ]; then
  echo "Error: benchmarking/tools/run.sh requires Python 3.9+ on the host." >&2
  exit 1
fi

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
eval_str=$("${PYTHON_BIN}" ${THIS_SCRIPT_DIR}/gen_runscript_vars.py "${BASH_SOURCE[0]}" "$@")
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
GPUS_FLAG=""
if [ "${GPUS}" != "none" ]; then
  GPUS_FLAG="--gpus=\"${GPUS}\""
fi

# --net=host allows the container to use the host's network stack, which Ray requires to
# communicate between the container and the host. When running multiple benchmarks in parallel,
# remove this flag so each container uses its own network namespace — this ensures each Ray
# cluster is confined to its own container and can use the same default ports without
# conflicting with other containers.
docker run \
  --rm \
  --net=host \
  --interactive \
  --tty \
  \
  ${GPUS_FLAG} \
  --memory=${CONTAINER_MEMORY_BYTES} \
  --shm-size=${SHM_SIZE_BYTES} \
  \
  ${VOLUME_MOUNTS} \
  \
  --env=NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  --env=IMAGE_DIGEST=${IMAGE_DIGEST} \
  --env=MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
  --env=SLACK_BOT_TOKEN=${SLACK_BOT_TOKEN} \
  --env=SLACK_CHANNEL_ID=${SLACK_CHANNEL_ID} \
  --env=GDRIVE_FOLDER_ID=${GDRIVE_FOLDER_ID} \
  --env=GDRIVE_SERVICE_ACCOUNT_FILE=${GDRIVE_SERVICE_ACCOUNT_FILE} \
  --env=CURATOR_BENCHMARKING_DEBUG=${CURATOR_BENCHMARKING_DEBUG} \
  --env=HOST_HOSTNAME=$(hostname) \
  --env=NVIDIA_API_KEY=${NVIDIA_API_KEY} \
  --env=AWS_ACCESS_KEY_ID \
  --env=AWS_SECRET_ACCESS_KEY \
  --env=AWS_DEFAULT_REGION \
  --env=AWS_ENDPOINT_URL \
  --env=S3_ENDPOINT_URL \
  --env=CURATOR_BENCHMARK_RAY_TMP_ROOT \
  --env=TMPDIR \
  --env=PYTORCH_CUDA_ALLOC_CONF \
  \
  ${BASH_ENTRYPOINT_OVERRIDE} \
  ${CURATOR_BENCHMARKING_IMAGE} \
    "${ENTRYPOINT_ARGS[@]}"

exit $?
