# ALM Pipeline Benchmark

A dedicated benchmark for measuring the performance of the ALM (Audio Language Model) data curation pipeline. The benchmark script lives at `benchmarking/scripts/alm_pipeline_benchmark.py` and runs through the full NeMo Curator benchmarking framework (Docker + Ray cluster + result collection).

## How It Works

The benchmark script:
1. Loads a JSONL manifest (supports cloud paths via fsspec)
2. Optionally multiplies entries with `--repeat-factor` for scale testing
3. Builds a Pipeline with ALMDataBuilderStage + ALMDataOverlapStage
4. Runs through XennaExecutor (or ray_data/ray_actors)
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
For CI/nightly runs, the benchmark is invoked via `benchmarking/tools/run.sh`
using the `alm_pipeline_xenna` entry. See `benchmarking/README.md` for details.

> **Remember** to re-enable the Slack sink (`enabled: true`) before pushing.

## Benchmark Configuration

The ALM benchmark entry is defined in `benchmarking/nightly-benchmark.yaml`:

```yaml
entries:
  - name: alm_pipeline_xenna
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
    requirements:
      - metric: is_success
        exact_value: true
      - metric: total_builder_windows
        min_value: 1
      - metric: total_filtered_windows
        min_value: 1
```

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

## Benchmark Results

Results from running on a single workstation:

**Machine specs:**
- CPU: Intel Core i9-9900KF @ 3.60GHz (8 cores / 16 threads)
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 3080 Ti 12 GB (not used by ALM stages)
- OS: Ubuntu 20.04, Linux 5.15

**Small scale (5 entries, sample fixture):**

| Metric | Value |
|--------|-------|
| Input entries | 5 |
| Output entries | 5 |
| Builder windows | 181 |
| Filtered windows | 25 |
| Total filtered duration | 3,035.50s |
| Execution time | 15.50s |
| Throughput (entries/sec) | 0.32 |

**Large scale (10,000 entries, repeat-factor=2000):**

| Metric | Value |
|--------|-------|
| Input entries | 10,000 |
| Output entries | 10,000 |
| Builder windows | 362,000 |
| Filtered windows | 50,000 |
| Total filtered duration | 6,071,000s |
| Execution time | 77.32s |
| Throughput (entries/sec) | 129.33 |
| Throughput (windows/sec) | 4,683.27 |

The pipeline scales well with XennaExecutor auto-allocating workers per stage via the CompositeStage reader (FilePartitioningStage + ALMManifestReaderStage). Throughput increases significantly at scale as the executor amortizes startup overhead.

## Output Files

The benchmark produces three files in `--benchmark-results-path`:

| File | Description |
|------|-------------|
| `params.json` | All pipeline parameters for reproducibility |
| `metrics.json` | `is_success`, `time_taken_s`, `throughput_entries_per_sec`, `throughput_windows_per_sec`, window counts, durations |
| `tasks.pkl` | Pickled task objects for `TaskPerfUtils` aggregation |
