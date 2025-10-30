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

DOCKER_IMAGE=${DOCKER_IMAGE:-nemo_curator_benchmarking:latest}
GPUS=${GPUS:-'"device=1"'}
HOST_CURATOR_DIR=${HOST_CURATOR_DIR:-"$(cd ${THIS_SCRIPT_DIR}/../.. && pwd)"}
CONTAINER_CURATOR_DIR="/opt/Curator"
HOST_PATHS_CONFIG_FILE=${HOST_PATHS_CONFIG_FILE:-$(realpath "${THIS_SCRIPT_DIR}/../config.yaml")}

################################################################################################################
# Get directory paths from ${HOST_PATHS_CONFIG_FILE}
get_dir_from_paths_config() {
    local dir_name=$1
    local dir_type=$2
    # TODO: Calling python is rather slow, consider requiring yq instead.
    python -c "import yaml; print(yaml.safe_load(open('${HOST_PATHS_CONFIG_FILE}'))['paths']['${dir_name}']['${dir_type}'].replace('\"',''))" 2>/dev/null
}
if [ -f "${HOST_PATHS_CONFIG_FILE}" ]; then
    HOST_RESULTS_DIR=${HOST_RESULTS_DIR:-$(get_dir_from_paths_config 'results_path' 'host')}
    CONTAINER_RESULTS_DIR=$(get_dir_from_paths_config 'results_path' 'container')
    HOST_ARTIFACTS_DIR=${HOST_ARTIFACTS_DIR:-$(get_dir_from_paths_config 'artifacts_path' 'host')}
    CONTAINER_ARTIFACTS_DIR=$(get_dir_from_paths_config 'artifacts_path' 'container')
    HOST_DATASETS_DIR=${HOST_DATASETS_DIR:-$(get_dir_from_paths_config 'datasets_path' 'host')}
    CONTAINER_DATASETS_DIR=$(get_dir_from_paths_config 'datasets_path' 'container')
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    # Show help and exit. Do this here to allow for help options to be passed to the container when other args are present.
    echo "Usage: $(basename "$0") [OPTIONS] [ARGS ...]"
    echo ""
    echo "Options:"
    echo "  --use-host-curator       Mount \$HOST_CURATOR_DIR into the container for benchmarking/debugging curator sources without rebuilding the image."
    echo "  --shell                  Start an interactive bash shell instead of running benchmarks. ARGS, if specified, will be passed to 'bash -c'."
    echo "                           For example: '--shell uv pip list | grep cugraph' will run 'uv pip list | grep cugraph' to display the version of cugraph installed in the container."
    echo "  -h, --help               Show this help message and exit."
    echo ""
    echo "ARGS, if specified, are passed to the container entrypoint, either the default benchmarking entrypoint or the --shell bash entrypoint."
    echo ""
    echo "Optional environment variables to override config and defaults:"
    echo "  HOST_PATHS_CONFIG_FILE    YAML file containing the mapping of container to host directory paths (using: ${HOST_PATHS_CONFIG_FILE})"
    echo "  GPUS                      Value for --gpus option to docker run (using: ${GPUS})"
    echo "  DOCKER_IMAGE              Docker image to use (using: ${DOCKER_IMAGE})"
    echo "  HOST_CURATOR_DIR          Curator repo path used with --use-host-curator (see above) (using: ${HOST_CURATOR_DIR})"
    echo "  HOST_DATASETS_DIR         Path to datasets on the host (using: ${HOST_DATASETS_DIR})"
    echo "  HOST_RESULTS_DIR          Results output directory on the host (using: ${HOST_RESULTS_DIR})"
    echo "  HOST_ARTIFACTS_DIR        Artifacts output directory on the host (using: ${HOST_ARTIFACTS_DIR})"
    exit 0
fi

if [ ! -d "${HOST_RESULTS_DIR}" ]; then
    echo "Error: Host results directory not found: \"${HOST_RESULTS_DIR}\". Ensure HOST_RESULTS_DIR is set or ${HOST_PATHS_CONFIG_FILE} is configured correctly."
    exit 1
fi
if [ ! -d "${HOST_ARTIFACTS_DIR}" ]; then
    echo "Error: Host artifacts directory not found: \"${HOST_ARTIFACTS_DIR}\". Ensure HOST_ARTIFACTS_DIR is set or ${HOST_PATHS_CONFIG_FILE} is configured correctly."
    exit 1
fi
if [ ! -d "${HOST_DATASETS_DIR}" ]; then
    echo "Error: Host datasets directory not found: \"${HOST_DATASETS_DIR}\". Ensure HOST_DATASETS_DIR is set or ${HOST_PATHS_CONFIG_FILE} is configured correctly."
    exit 1
fi

CURATOR_SOURCE_DIR_OVERRIDE=""
BASH_ENTRYPOINT_OVERRIDE=""
ENTRYPOINT_ARGS=()
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --use-host-curator)
            CURATOR_SOURCE_DIR_OVERRIDE="--volume ${HOST_CURATOR_DIR}:${CONTAINER_CURATOR_DIR}"
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

IMAGE_DIGEST=$(docker image inspect ${DOCKER_IMAGE} --format '{{.Digest}}' 2>/dev/null)
if [ -z "${IMAGE_DIGEST}" ] || [ "${IMAGE_DIGEST}" = "<none>" ]; then
    # Use the image ID as a fallback
    IMAGE_DIGEST=$(docker image inspect ${DOCKER_IMAGE} --format '{{.ID}}' 2>/dev/null)
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
  --volume ${HOST_DATASETS_DIR}:${CONTAINER_DATASETS_DIR} \
  --volume ${HOST_RESULTS_DIR}:${CONTAINER_RESULTS_DIR} \
  --volume ${HOST_ARTIFACTS_DIR}:${CONTAINER_ARTIFACTS_DIR} \
  \
  --env=IMAGE_DIGEST=${IMAGE_DIGEST} \
  --env=MLFLOW_TRACKING_URI=blank \
  --env=SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL} \
  --env=GDRIVE_FOLDER_ID=${GDRIVE_FOLDER_ID} \
  --env=GDRIVE_SERVICE_ACCOUNT_FILE=${GDRIVE_SERVICE_ACCOUNT_FILE} \
  \
  ${BASH_ENTRYPOINT_OVERRIDE} \
  ${DOCKER_IMAGE} \
    "${ENTRYPOINT_ARGS[@]}"

exit $?
