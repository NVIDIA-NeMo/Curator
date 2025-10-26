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

# Assume this script is in the <repo_root>/benchmarking/tools directory
THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS=${GPUS:-'"device=1"'}

DOCKER_IMAGE=${DOCKER_IMAGE:-nemo_curator_benchmarking:latest}

# Note: The CONTAINER_* env vars will also be set in the container so
# they can be used for specifying paths in YAML config files.
LOCAL_CURATOR_DIR=${LOCAL_CURATOR_DIR:-"$(cd ${THIS_SCRIPT_DIR}/../.. && pwd)"}
CONTAINER_CURATOR_DIR=/opt/Curator

LOCAL_DATASETS_DIR=${LOCAL_DATASETS_DIR:-"/datasets/curator"}
CONTAINER_DATASETS_DIR=/data/datasets

LOCAL_RESULTS_DIR=${LOCAL_RESULTS_DIR:-"/home/rratzel/tmp/curator_benchmark_results/results"}
CONTAINER_RESULTS_DIR=/data/benchmarking/results

LOCAL_ARTIFACTS_DIR=${LOCAL_ARTIFACTS_DIR:-"/home/rratzel/tmp/curator_benchmark_results/artifacts"}
CONTAINER_ARTIFACTS_DIR=/data/benchmarking/artifacts


################################################################################################################
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    # Show help and exit. Do this here to allow for help options to be passed to the container when other args are present.
    echo "Usage: $(basename "$0") [OPTIONS] [ARGS ...]"
    echo ""
    echo "Options:"
    echo "  --use-local-curator      Mount \$LOCAL_CURATOR_DIR (see below) into the container for benchmarking/debugging local sources without rebuilding the image."
    echo "  --shell                  Start an interactive bash shell instead of running benchmarks. ARGS, if specified, will be passed to 'bash -c'."
    echo "                           For example: '--shell uv pip list | grep cugraph' will run 'uv pip list | grep cugraph' to display the version of cugraph installed in the container."
    echo "  -h, --help               Show this help message and exit."
    echo ""
    echo "ARGS, if specified, are passed to the container entrypoint, either the default benchmarking entrypoint or the --shell bash entrypoint."
    echo ""
    echo "Environment variables:"
    echo "  GPUS                     --gpus parameter for docker (using: ${GPUS})"
    echo "  DOCKER_IMAGE             Docker image to use (using: ${DOCKER_IMAGE})"
    echo "  LOCAL_CURATOR_DIR        Local Curator repo path (using: ${LOCAL_CURATOR_DIR})"
    echo "  LOCAL_DATASETS_DIR       Path to datasets on the host (using: ${LOCAL_DATASETS_DIR})"
    echo "  LOCAL_RESULTS_DIR        Results output directory on the host (using: ${LOCAL_RESULTS_DIR})"
    echo "  LOCAL_ARTIFACTS_DIR      Artifacts output directory on the host (using: ${LOCAL_ARTIFACTS_DIR})"
    exit 0
fi

CURATOR_SOURCE_DIR_OVERRIDE=""
BASH_ENTRYPOINT_OVERRIDE=""
ENTRYPOINT_ARGS=()
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --use-local-curator)
            CURATOR_SOURCE_DIR_OVERRIDE="--volume ${LOCAL_CURATOR_DIR}:${CONTAINER_CURATOR_DIR}"
            shift
            ;;
        --shell)
            BASH_ENTRYPOINT_OVERRIDE="--entrypoint=bash"
            shift
            ;;
        *)
            # unknown option, pass as-is to the entrypoint
            ENTRYPOINT_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -n "${BASH_ENTRYPOINT_OVERRIDE}" ] && [ "${#ENTRYPOINT_ARGS[@]}" -gt 0 ]; then
    # Add arguments as a single string to the entrypoint after -c
    # so ENTRYPOINT_ARGS is always an array with two items.
    # ex. ["-c", "arg1 arg2 arg3"] -> bash -c "arg1 arg2 arg3"
    ENTRYPOINT_ARGS=("-c" "$(printf "%s " "${ENTRYPOINT_ARGS[@]}")")
fi


################################################################################################################
docker run \
  --rm \
  --interactive \
  --tty \
  \
  --gpus=${GPUS} \
  \
  ${CURATOR_SOURCE_DIR_OVERRIDE} \
  --volume ${LOCAL_DATASETS_DIR}:${CONTAINER_DATASETS_DIR} \
  --volume ${LOCAL_RESULTS_DIR}:${CONTAINER_RESULTS_DIR} \
  --volume ${LOCAL_ARTIFACTS_DIR}:${CONTAINER_ARTIFACTS_DIR} \
  --env=CONTAINER_DATASETS_DIR=${CONTAINER_DATASETS_DIR} \
  --env=CONTAINER_RESULTS_DIR=${CONTAINER_RESULTS_DIR} \
  --env=CONTAINER_ARTIFACTS_DIR=${CONTAINER_ARTIFACTS_DIR} \
  \
  --env=MLFLOW_TRACKING_URI=blank \
  --env=SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL} \
  \
  ${BASH_ENTRYPOINT_OVERRIDE} \
  ${DOCKER_IMAGE} \
    "${ENTRYPOINT_ARGS[@]}"

exit $?
