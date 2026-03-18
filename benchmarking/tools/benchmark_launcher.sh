#!/bin/bash
set -ex

mkdir -p "/tmp/curator/results/${BRANCH_NAME}"

cd /opt/Curator
uv pip install GitPython pynvml pyyaml rich

python benchmarking/run.py \
  --config benchmarking/nightly-benchmark.yaml \
  --config /opt/Curator/benchmarking/test-paths.yaml \
  --session-name "benchmark_run_${CI_PIPELINE_ID}" \
  --entries "${ENTRY_NAME}"
