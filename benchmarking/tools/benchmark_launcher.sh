#!/bin/bash
set -ex

mkdir -p "/tmp/curator/results/${BRANCH_NAME}"

# Install benchmarking runner dependencies not included in the main Curator image
# (these are normally added by benchmarking/Dockerfile which CI does not use)
cd /opt/Curator
uv pip install GitPython pynvml pyyaml rich

# run.py supports --config multiple times; later configs override earlier ones
python benchmarking/run.py \
  --config benchmarking/nightly-benchmark.yaml \
  --config /opt/Curator/benchmarking/test-paths.yaml \
  --session-name "benchmark_run_${CI_PIPELINE_ID}" \
  --entries "${ENTRY_NAME}"
