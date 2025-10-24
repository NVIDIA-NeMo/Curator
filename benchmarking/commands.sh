#!/bin/bash
docker run \
  --gpus='"device=1"' \
  --rm \
  -it \
  --volume /raid/rratzel/nemo_curator_data:/data \
  --volume /raid/rratzel/curator/benchmarking:/opt/Curator/benchmarking \
  --env=MLFLOW_TRACKING_URI=dsa \
  \
  bench \
    --config=/opt/Curator/benchmarking/config.yaml \
    --config=/opt/Curator/benchmarking/paths.yaml

exit $?



python ./scripts/common_crawl_benchmark.py \
      --benchmark-results-path /tmp/session_entry_dir/benchmark_results \
      --download_path /tmp/session_entry_dir/scratch/downloads \
      --output_path /tmp/session_entry_dir/scratch/output \
      --output_format parquet \
      --crawl_type main \
      --start_snapshot 2023-01 \
      --end_snapshot 2023-10 \
      --html_extraction justext \
      --url_limit 10 \
      --add_filename_column \
      --executor ray_data \
      --ray_data_cast_as_actor
