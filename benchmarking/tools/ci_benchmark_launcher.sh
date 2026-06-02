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

# Optional: cap /dev/shm to CURATOR_SHM_SIZE_BYTES bytes for cross-host
# (e.g. A100 vs H100) comparison runs. Unset → no remount, leaving the
# container's default /dev/shm size (host tmpfs) in place.
if [ -n "${CURATOR_SHM_SIZE_BYTES:-}" ]; then
  if mount -o remount,size="${CURATOR_SHM_SIZE_BYTES}" /dev/shm 2>/dev/null; then
    echo "[ci_benchmark_launcher] /dev/shm remounted to ${CURATOR_SHM_SIZE_BYTES} bytes"
  else
    echo "[ci_benchmark_launcher] WARNING: failed to remount /dev/shm to ${CURATOR_SHM_SIZE_BYTES} bytes (insufficient caps?); continuing"
  fi
fi

# Optional: poll all GPUs every CURATOR_GPU_POLL_INTERVAL_S seconds and dump a
# CSV per entry. Useful for verifying CUDA_VISIBLE_DEVICES is actually honored
# by Ray/Xenna (so we can detect post-run if unmasked GPUs were touched).
# Unset → no polling.
NVSMI_PID=""
if [ -n "${CURATOR_GPU_POLL_INTERVAL_S:-}" ] && [ "${CURATOR_GPU_POLL_INTERVAL_S}" -gt 0 ]; then
    GPU_UTIL_DIR="/tmp/curator/results/${BRANCH_NAME}/benchmark_run_${CI_PIPELINE_ID}"
    mkdir -p "${GPU_UTIL_DIR}"
    GPU_UTIL_CSV="${GPU_UTIL_DIR}/gpu_util_${ENTRY_NAME}_slurm${SLURM_JOB_ID:-pid$$}.csv"
    echo "[ci_benchmark_launcher] Polling GPUs every ${CURATOR_GPU_POLL_INTERVAL_S}s -> ${GPU_UTIL_CSV}"
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
        --format=csv,nounits \
        -l "${CURATOR_GPU_POLL_INTERVAL_S}" > "${GPU_UTIL_CSV}" 2>&1 &
    NVSMI_PID=$!
    trap '[ -n "${NVSMI_PID}" ] && kill "${NVSMI_PID}" 2>/dev/null || true' EXIT
fi

cd /opt/Curator
uv pip install GitPython pynvml pyyaml rich

python benchmarking/run.py \
  --config /opt/Curator/benchmarking/nightly-benchmark.yaml \
  --config /opt/Curator/benchmarking/test-paths.yaml \
  --session-name "benchmark_run_${CI_PIPELINE_ID}" \
  --entries-exact "${ENTRY_NAME}"
