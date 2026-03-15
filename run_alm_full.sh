#!/bin/bash
docker run --rm --net=host --shm-size=8g \
  -v $(pwd):/opt/Curator \
  -v /home/aaftabv/curator/data:/data \
  --entrypoint bash nemo_curator_benchmarking:latest \
  -c "cd /opt/Curator && python benchmarking/scripts/alm_pipeline_benchmark.py \
    --benchmark-results-path=/data/alm_bench_full468 \
    --input-manifest=/data/fused_ia_top3.jsonl \
    --executor=xenna \
    --target-window-duration=120.0 \
    --tolerance=0.1 \
    --min-sample-rate=16000 \
    --min-bandwidth=8000 \
    --min-speakers=2 \
    --max-speakers=5 \
    --overlap-percentage=50"
