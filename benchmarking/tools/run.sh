#!/bin/bash

# FIXME: Look into making use cases that rely on docker volume mounts formally supported
#   by the benchmarking framework. For example, the notion of local and container dir mappings
#   could be in the same config YAML, and the tool would then automate launching the container.
#   This would make make it easier than coordinating paths in both the YAML for datasets,
#   results, etc. and this script to ensure the volume mounts exist.
LOCAL_CURATOR_DIR=/home/rratzel/Projects/curator
CONTAINER_CURATOR_DIR=/opt/Curator
LOCAL_DATASETS_DIR=/datasets/curator
CONTAINER_DATASETS_DIR=/datasets
LOCAL_RESULTS_DIR=/home/rratzel/tmp/curator_benchmark_results/results
CONTAINER_RESULTS_DIR=/benchmarking/results
LOCAL_ARTIFACTS_DIR=/home/rratzel/tmp/curator_benchmark_results/artifacts
CONTAINER_ARTIFACTS_DIR=/benchmarking/artifacts

docker run \
  --gpus='"device=1"' \
  --rm \
  -it \
  --volume $LOCAL_DATASETS_DIR:$CONTAINER_DATASETS_DIR \
  --volume $LOCAL_RESULTS_DIR:$CONTAINER_RESULTS_DIR \
  --volume $LOCAL_ARTIFACTS_DIR:$CONTAINER_ARTIFACTS_DIR \
  --volume ${LOCAL_CURATOR_DIR}/benchmarking:${CONTAINER_CURATOR_DIR}/benchmarking \
  --env=MLFLOW_TRACKING_URI=blank \
  --env=SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL} \
  \
  bench \
    --config=${CONTAINER_CURATOR_DIR}/benchmarking/rratzel-ws1--config.yaml \
    --config=${CONTAINER_CURATOR_DIR}/benchmarking/container_paths.yaml

exit $?
