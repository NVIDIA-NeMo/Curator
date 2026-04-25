# ALM Pipeline Benchmark

A dedicated benchmark for measuring the performance of the ALM (Audio Language Model) data curation pipeline. The benchmark script lives at `benchmarking/scripts/alm_pipeline_benchmark.py` and runs through the full NeMo Curator benchmarking framework (Docker + Ray cluster + result collection).

## How It Works

The benchmark script:
1. Loads a JSONL manifest via `ALMManifestReader` (CompositeStage: FilePartitioningStage + ALMManifestReaderStage)
2. Optionally multiplies entries with `--repeat-factor` via `_RepeatEntriesStage`
3. Runs `ALMDataBuilderStage` (windowing) + `ALMDataOverlapStage` (filtering)
4. Executes through XennaExecutor, RayDataExecutor, or RayActorPoolExecutor
5. Writes `params.json`, `metrics.json`, and `tasks.pkl` for the framework

## Running Benchmarks

The benchmarking framework is designed to run inside Docker. The benchmarking image
installs additional dependencies (`GitPython`, `pynvml`, `rich`, `slack_sdk`, etc.)
that are not part of `nemo_curator` itself. This applies to all benchmarks in the
repository, not just ALM.

**Prerequisites:**
- Docker with NVIDIA container toolkit
- NeMo Curator repository checked out

**Step 1: Build the benchmarking Docker image (one-time):**

```bash
cd /path/to/Curator
bash benchmarking/tools/build_docker.sh --tag-as-latest
```

**Step 2: Disable the Slack sink for local runs.**

The shared `benchmarking/nightly-benchmark.yaml` has the Slack sink enabled, which
requires valid `SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` credentials (used in CI).
For local testing, temporarily disable it:

```yaml
sinks:
  - name: slack
    enabled: false   # <-- change from true to false
    live_updates: true
    channel_id: ${SLACK_CHANNEL_ID}
    default_metrics: ["exec_time_s"]
```

**Step 3: Run the ALM benchmark:**

The `--config` flag reads parameters directly from the `alm_pipeline_xenna`
entry in `benchmarking/nightly-benchmark.yaml`:

```bash
docker run --rm --net=host --shm-size=8g \
  -v $(pwd):/opt/Curator \
  --entrypoint bash nemo_curator_benchmarking:latest \
  -c "cd /opt/Curator && python benchmarking/scripts/alm_pipeline_benchmark.py \
    --config benchmarking/nightly-benchmark.yaml"
```

The ALM pipeline is CPU-only so no `--gpus` flag is needed.

**Running locally (without Docker):**

```bash
cd /path/to/Curator
python benchmarking/scripts/alm_pipeline_benchmark.py \
  --benchmark-results-path /tmp/alm_bench \
  --input-manifest tests/fixtures/audio/alm/sample_input.jsonl
```

Or with `--config` to read parameters from the nightly YAML:

```bash
python benchmarking/scripts/alm_pipeline_benchmark.py \
  --config benchmarking/nightly-benchmark.yaml
```

For CI/nightly runs, the benchmark is invoked via `benchmarking/tools/run.sh`
using the `alm_pipeline_xenna` and `alm_pipeline_ray_data` entries.
See `benchmarking/README.md` for details.

> **Remember** to re-enable the Slack sink (`enabled: true`) before pushing.

## Benchmark Configuration

Two ALM benchmark entries are defined in `benchmarking/nightly-benchmark.yaml` —
one for each backend:

**`alm_pipeline_xenna`** (default backend):

```yaml
  - name: alm_pipeline_xenna
    enabled: true
    script: alm_pipeline_benchmark.py
    args: >-
      --benchmark-results-path={session_entry_dir}
      --input-manifest={curator_repo_dir}/tests/fixtures/audio/alm/sample_input.jsonl
      --executor=xenna
      --target-window-duration=120.0
      --tolerance=0.1
      --min-sample-rate=16000
      --min-bandwidth=8000
      --min-speakers=2
      --max-speakers=5
      --overlap-percentage=50
      --repeat-factor=2000
    timeout_s: 600
    sink_data:
      - name: slack
        ping_on_failure:
          - U03C41SNADV  # Aaftab V
    ray:
      num_cpus: 8
      num_gpus: 0
      enable_object_spilling: false
    requirements:
      - metric: is_success
        exact_value: true
      - metric: total_builder_windows
        min_value: 1
      - metric: total_filtered_windows
        min_value: 1
```

**`alm_pipeline_ray_data`** (Ray Data backend):

```yaml
  - name: alm_pipeline_ray_data
    enabled: true
    script: alm_pipeline_benchmark.py
    args: >-
      --benchmark-results-path={session_entry_dir}
      --input-manifest={curator_repo_dir}/tests/fixtures/audio/alm/sample_input.jsonl
      --executor=ray_data
      --target-window-duration=120.0
      --tolerance=0.1
      --min-sample-rate=16000
      --min-bandwidth=8000
      --min-speakers=2
      --max-speakers=5
      --overlap-percentage=50
      --repeat-factor=2000
    timeout_s: 600
    sink_data:
      - name: slack
        ping_on_failure:
          - U03C41SNADV  # Aaftab V
    ray:
      num_cpus: 8
      num_gpus: 0
      enable_object_spilling: false
    requirements:
      - metric: is_success
        exact_value: true
      - metric: total_builder_windows
        min_value: 1
      - metric: total_filtered_windows
        min_value: 1
```

Both entries use the same pipeline parameters (repeat-factor=2000, 120s windows,
50% overlap) but differ only in `--executor`.  The ALM pipeline is CPU-only
(`num_gpus: 0`).

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--benchmark-results-path` | Required | Directory for output files |
| `--input-manifest` | Required | Path to JSONL manifest |
| `--executor` | `xenna` | `xenna`, `ray_data`, or `ray_actors` |
| `--repeat-factor` | `1` | Multiply manifest entries for scale testing |
| `--target-window-duration` | `120.0` | Target window duration (seconds) |
| `--tolerance` | `0.1` | Window duration tolerance fraction |
| `--min-sample-rate` | `16000` | Minimum audio sample rate |
| `--min-bandwidth` | `8000` | Minimum segment bandwidth |
| `--min-speakers` | `2` | Minimum speakers per window |
| `--max-speakers` | `5` | Maximum speakers per window |
| `--overlap-percentage` | `50` | Overlap filter percentage (0-100) |
| `--execution-mode` | `None` | Xenna execution mode: `streaming` (default) or `batch` |

## Benchmark Results

Results from running on a single workstation. All scripts use `RayClient` for
managed Ray cluster lifecycle (start/stop, port allocation, dashboard).

**Machine specs:**
- CPU: Intel Core i9-9900KF @ 3.60GHz (8 cores / 16 threads)
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 3080 Ti 12 GB (not used by ALM stages)
- OS: Ubuntu 20.04, Linux 5.15

**Small scale (5 entries, sample fixture):**

| Metric | Xenna (streaming) | Xenna (batch) | Ray Data |
|--------|-------------------|---------------|----------|
| Input entries | 5 | 5 | 5 |
| Output entries | 5 | 5 | 5 |
| Builder windows | 181 | 181 | 181 |
| Filtered windows | 25 | 25 | 25 |
| Total filtered duration | 3,035.50s | 3,035.50s | 3,035.50s |
| Execution time | 22.72s | 28.46s | 8.84s |
| Throughput (entries/sec) | 0.22 | 0.18 | 0.57 |
| Throughput (windows/sec) | 7.97 | 6.36 | 20.48 |

**Large scale (10,000 entries, repeat-factor=2000):**

| Metric | Xenna (streaming) | Xenna (batch) | Ray Data |
|--------|-------------------|---------------|----------|
| Input entries | 10,000 | 10,000 | 10,000 |
| Output entries | 10,000 | 10,000 | 10,000 |
| Builder windows | 362,000 | 362,000 | 362,000 |
| Filtered windows | 50,000 | 50,000 | 50,000 |
| Total filtered duration | 6,071,000s | 6,071,000s | 6,071,000s |
| Execution time | 94.79s | 63.65s | 27.65s |
| Throughput (entries/sec) | 105.50 | 157.10 | 361.70 |
| Throughput (windows/sec) | 3,819.09 | 5,687.17 | 13,093.50 |

All three configurations produce identical results. Key findings:

- **Small scale:** Xenna batch mode is slower than streaming due to fixed overhead of
  placing all inputs into the Ray object store up front. At small scale, the streaming
  autoscaler finishes before its overhead becomes significant.
- **Large scale:** Xenna batch mode is ~1.5x faster than streaming because the
  autoscaler polling (every 180s) and per-stage actor lifecycle add up over many tasks.
- **Ray Data** is consistently the fastest (~3x vs streaming) for this CPU-only pipeline
  because it avoids per-stage actor creation/teardown entirely.
- For GPU-heavy pipelines, Xenna streaming mode is recommended as its autoscaler
  dynamically adjusts GPU worker counts.

See [AUDIO_PROFILING.md](AUDIO_PROFILING.md) for per-stage breakdown.

The `repeat-factor` multiplies entries in-memory after reading (via `_RepeatEntriesStage`), so the manifest file is read only once. The pipeline scales well with both executors via the CompositeStage reader (FilePartitioningStage + ALMManifestReaderStage).

## Output Files

The benchmark produces three files in `--benchmark-results-path`:

| File | Description |
|------|-------------|
| `params.json` | All pipeline parameters for reproducibility |
| `metrics.json` | `is_success`, `time_taken_s`, `throughput_entries_per_sec`, `throughput_windows_per_sec`, window counts, durations |
| `tasks.pkl` | Pickled task objects for `TaskPerfUtils` aggregation |
