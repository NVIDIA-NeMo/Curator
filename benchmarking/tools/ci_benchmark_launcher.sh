#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

set -ex

mkdir -p "/tmp/curator/results/${BRANCH_NAME}"

# Install lynx unconditionally. The math/* benchmarks shell out to lynx for HTML
# extraction. lynx is GPL-licensed so we deliberately do not bake it into the
# redistributable Curator image; instead it is installed transiently at CI run
# time inside the existing benchmark container, used during the run, and
# discarded with the container.
apt-get update -qq && apt-get install -y --no-install-recommends lynx

# ffmpeg not in image (CVE removal); install at runtime per modality
if [[ "${ENTRY_NAME}" == audio_* ]]; then
    apt-get install -y --no-install-recommends ffmpeg
elif [[ "${ENTRY_NAME}" == video_* ]]; then
    bash /opt/Curator/docker/common/install_ffmpeg.sh
fi

cd /opt/Curator
uv pip install GitPython pynvml pyyaml rich

# Session name resolution:
#   - If NEMO_CI_SESSION_NAME is set by the generated benchmark pipeline, use it
#     verbatim so every benchmark job writes to the same session directory.
#   - Else fall back to the legacy benchmark_run_<pipeline-id> name so existing
#     manual launches via launch_pipeline.py keep producing the same paths.
if [ -n "${NEMO_CI_SESSION_NAME:-}" ]; then
    SESSION_NAME="${NEMO_CI_SESSION_NAME}"
else
    SESSION_NAME="benchmark_run_${CI_PIPELINE_ID}"
fi

# Compose viewer URL only when a viewer host was provided by the launcher.
# Host path used so the viewer can read results without container-mount knowledge.
VIEWER_URL=""
if [ -n "${NEMO_CI_VIEWER_HOST:-}" ]; then
    RESULTS_HOST_DIR="${DEFAULT_CLUSTER_DIR}/curator_ci/results/${BRANCH_NAME}/${SESSION_NAME}"
    ENC_DIR=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "${RESULTS_HOST_DIR}")
    ENC_RUN=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "${SESSION_NAME}")
    VIEWER_URL="http://${NEMO_CI_VIEWER_HOST}/run-viewer?dir=${ENC_DIR}&run=${ENC_RUN}"
fi

python benchmarking/run.py \
  --config /opt/Curator/benchmarking/nightly-benchmark.yaml \
  --config /opt/Curator/benchmarking/test-paths.yaml \
  --session-name "${SESSION_NAME}" \
  ${VIEWER_URL:+--viewer-url "${VIEWER_URL}"} \
  ${NEMO_CI_RUN_REASON:+--reason "${NEMO_CI_RUN_REASON}"} \
  --entries-exact "${ENTRY_NAME}"
