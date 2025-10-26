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

# Assume this script is in the <repo_root>benchmarking/tools directory
THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_DIR="$(cd ${THIS_SCRIPT_DIR}/../.. && pwd)"

GPUS=${GPUS:-'"device=1"'}

DOCKER_IMAGE=${DOCKER_IMAGE:-nemo_curator_benchmarking:latest}

LOCAL_CURATOR_DIR=${CURATOR_DIR}
CONTAINER_CURATOR_DIR=/opt/Curator

LOCAL_DATASETS_DIR=/datasets/curator
CONTAINER_DATASETS_DIR=/data/datasets

LOCAL_RESULTS_DIR=/home/rratzel/tmp/curator_benchmark_results/results
CONTAINER_RESULTS_DIR=/data/benchmarking/results

LOCAL_ARTIFACTS_DIR=/home/rratzel/tmp/curator_benchmark_results/artifacts
CONTAINER_ARTIFACTS_DIR=/data/benchmarking/artifacts

DOCKER_ENTRYPOINT_OVERRIDE=""
CONFIG_FILE=${CONTAINER_CURATOR_DIR}/benchmarking/config.yaml

EXTRA_ARGS=""
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --config)
            CONFIG_FILE="$2"
            shift # past argument
            shift # past value
            ;;
        --shell)
            DOCKER_ENTRYPOINT_OVERRIDE="--entrypoint=bash"
            shift
            ;;
        *)
            # unknown option, pass as-is to the entrypoint
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done
# Add the config file only if the default entrypoint is used
if [ -n "${DOCKER_ENTRYPOINT_OVERRIDE}" ]; then
    ENTRYPOINT_ARGS="${EXTRA_ARGS}"
else
    ENTRYPOINT_ARGS="--config=${CONFIG_FILE} ${EXTRA_ARGS}"
fi

########################################################
docker run \
  --rm \
  --interactive \
  --tty \
  \
  --gpus=${GPUS} \
  \
  --volume ${LOCAL_CURATOR_DIR}:${CONTAINER_CURATOR_DIR} \
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
  ${DOCKER_ENTRYPOINT_OVERRIDE} \
  ${DOCKER_IMAGE} \
    ${ENTRYPOINT_ARGS}

exit $?
