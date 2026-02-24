# ALM (Audio Language Model) Data Pipeline

This tutorial demonstrates how to create training windows from audio segments for Audio Language Model (ALM) training using NeMo Curator.

## Overview

The ALM pipeline processes audio manifests containing diarized segments and creates training windows with the following filters:

- **Sample rate**: Minimum 16kHz
- **Bandwidth**: Minimum 8kHz per segment
- **Window duration**: Target 120 seconds (±10% tolerance)
- **Speaker count**: 2-5 speakers per window
- **Overlap filtering**: Remove highly overlapping windows

### Pipeline Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Diarized Audio │───▶│  ALM Pipeline   │───▶│   Downstream    │───▶│  Sharded Data   │
│    Manifests    │    │  (this stage)   │    │   Processors    │    │  for Training   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
      input.jsonl          output.jsonl           (future stages)        ready for ALM
```

The output JSONL from this pipeline is consumed by downstream processors for additional processing (e.g., audio slicing, feature extraction, data augmentation). At the end of the full pipeline, the output will be sharded data ready for training Audio Language Models.

## Installation

From the Curator repository root:

```bash
uv sync --extra audio_cpu
source .venv/bin/activate
```

This creates a `.venv` with all base, dev, test, and audio dependencies resolved
from the lockfile. If you don't have `uv`, you can fall back to pip:

```bash
pip install -e ".[audio_cpu]"
```

## Sample Data

Sample data is located in `tests/fixtures/audio/alm/` for use in both testing and tutorials:

```
tests/fixtures/audio/alm/
└── sample_input.jsonl        # 5 sample audio manifests with diarized segments

tutorials/audio/alm/
├── main.py                   # Pipeline runner (YAML-driven)
├── pipeline.yaml             # Pipeline configuration
└── README.md                 # This file
```

The sample input contains 5 audio manifest entries with:
- Various sample rates (16kHz, 22kHz, 44kHz, 48kHz)
- 30+ segments per entry with speaker diarization
- Bandwidth metrics for quality filtering
- Multiple speakers (2-4 per conversation)

## Quick Start

Run the pipeline on the included sample data (from Curator repo root):

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  manifest_path=tests/fixtures/audio/alm/sample_input.jsonl
```

Expected output:
```
PIPELINE COMPLETE
==================================================
  Output entries: 5
  [alm_manifest_reader]
    process_time: mean=0.0030s, total=0.01s
    items_processed: 0
  [alm_data_builder]
    process_time: mean=0.0015s, total=0.01s
    items_processed: 5
    windows_created: 181
  [alm_data_overlap]
    process_time: mean=0.0004s, total=0.00s
    items_processed: 5
    output_windows (after overlap): 25
    filtered_audio_duration: 3035.5s
  [alm_manifest_writer]
    process_time: mean=0.0001s, total=0.00s
    items_processed: 5
```

## Using Custom Data

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  manifest_path=/path/to/your/data.jsonl \
  output_dir=./my_output
```

## Configuration

All parameters are defined in `pipeline.yaml`. Override from command line:

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_dir=./custom_output \
  stages.1.min_speakers=3 \
  stages.1.max_speakers=6 \
  stages.2.overlap_percentage=30
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `manifest_path` | Path to input JSONL manifest | Required |
| `output_dir` | Directory for output files | `./alm_output` |
| `stages.1.target_window_duration` | Target window duration (seconds) | `120.0` |
| `stages.1.tolerance` | Duration tolerance (e.g., 0.1 = ±10%) | `0.1` |
| `stages.1.min_sample_rate` | Minimum sample rate (Hz) | `16000` |
| `stages.1.min_bandwidth` | Minimum bandwidth (Hz) | `8000` |
| `stages.1.min_speakers` | Minimum speakers per window | `2` |
| `stages.1.max_speakers` | Maximum speakers per window | `5` |
| `stages.2.overlap_percentage` | Overlap threshold 0-100 | `50` |

### Override Notes

Match indices in `stages` list in `pipeline.yaml`:
- `stages.0.*`: ALMManifestReaderStage parameters
- `stages.1.*`: ALMDataBuilderStage parameters
- `stages.2.*`: ALMDataOverlapStage parameters

## Input Format

The input manifest should be a JSONL file where each line contains:

```json
{
  "audio_filepath": "/path/to/audio.wav",
  "audio_sample_rate": 16000,
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "speaker": "speaker_0",
      "text": "Hello, how are you?",
      "words": [{"word": "Hello", "start": 0.0, "end": 0.5}, ...],
      "metrics": {"bandwidth": 8000}
    },
    ...
  ]
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio_filepath` | string | Path to audio file |
| `audio_sample_rate` | int | Sample rate in Hz |
| `segments` | list | List of diarized segments |
| `segments[].start` | float | Segment start time (seconds) |
| `segments[].end` | float | Segment end time (seconds) |
| `segments[].speaker` | string | Speaker identifier |
| `segments[].metrics.bandwidth` | int | Segment bandwidth in Hz |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `segments[].text` | string | Transcription text |
| `segments[].words` | list | Word-level timestamps |

## Output Format

Results are written as JSONL to `${output_dir}/alm_output.jsonl`. Each line contains:

```json
{
  "audio_filepath": "/path/to/audio.wav",
  "windows": [...],
  "filtered_windows": [
    {
      "segments": [
        {"start": 10.0, "end": 15.2, "speaker": "speaker_0", "text": "..."},
        {"start": 15.5, "end": 22.1, "speaker": "speaker_1", "text": "..."},
        ...
      ],
      "speaker_durations": [45.2, 38.1, 22.5, 14.2, 0.0]
    }
  ],
  "filtered_dur": 120.5,
  "filtered_dur_list": [120.5],
  "stats": {
    "total_segments": 150,
    "total_dur": 3600.0,
    "lost_bw": 5,
    "lost_sr": 0,
    "lost_spk": 12,
    "lost_win": 8
  },
  "truncation_events": 3
}
```

### Output Fields

| Field | Description |
|-------|-------------|
| `windows` | All valid windows from builder stage |
| `filtered_windows` | Windows after overlap filtering |
| `filtered_dur` | Total duration of filtered windows |
| `filtered_dur_list` | Duration of each filtered window |
| `stats` | Processing statistics and loss reasons |
| `truncation_events` | Count of segment truncations |

## Pipeline Stages

### Stage 1: ALMDataBuilderStage

Creates training windows from audio segments.

**Processing Logic:**
1. Check audio sample rate (skip if < min_sample_rate)
2. For each segment as potential window start:
   - Check bandwidth requirement
   - Build window by adding consecutive segments
   - Apply truncation if window exceeds max duration
   - Validate speaker count (2-5 speakers)
   - Check window duration (target ± tolerance)
3. Create window with segments and speaker durations

**Statistics Tracked:**
- `lost_bw`: Segments lost due to low bandwidth
- `lost_sr`: Segments lost due to low sample rate
- `lost_spk`: Segments lost due to speaker count
- `lost_win`: Segments lost due to window constraints

### Stage 2: ALMDataOverlapStage

Filters windows based on overlap ratio.

**Processing Logic:**
1. Calculate timestamps for all windows
2. For each pair of overlapping windows:
   - Calculate overlap ratio
   - If overlap exceeds threshold, keep window closer to target duration
3. Return filtered windows

**Parameters:**
- `overlap_percentage=0`: Aggressive filtering (remove any overlap)
- `overlap_percentage=50`: Moderate filtering
- `overlap_percentage=100`: Permissive (keep all windows)

## Customization Examples

### Adjusting Window Duration

For shorter windows (e.g., 60 seconds):

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  manifest_path=tests/fixtures/audio/alm/sample_input.jsonl \
  stages.1.target_window_duration=60 \
  stages.1.tolerance=0.15
```

### Stricter Speaker Requirements

For exactly 2-3 speakers:

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  manifest_path=tests/fixtures/audio/alm/sample_input.jsonl \
  stages.1.min_speakers=2 \
  stages.1.max_speakers=3
```

### Aggressive Overlap Filtering

Remove all overlapping windows:

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  manifest_path=tests/fixtures/audio/alm/sample_input.jsonl \
  stages.2.overlap_percentage=0
```

## Benchmarking

A dedicated benchmark script is provided at `benchmarking/scripts/alm_pipeline_benchmark.py` for measuring pipeline performance at scale and catching regressions. It runs through the full NeMo Curator benchmarking framework (Docker + Ray cluster + result collection).

### How It Works

The benchmark script:
1. Loads a JSONL manifest (supports cloud paths via fsspec)
2. Optionally multiplies entries with `--repeat-factor` for scale testing
3. Builds a Pipeline with ALMDataBuilderStage + ALMDataOverlapStage
4. Runs through XennaExecutor (or ray_data/ray_actors)
5. Writes `params.json`, `metrics.json`, and `tasks.pkl` for the framework

### Running Benchmarks

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

### Benchmark Configuration

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

### CLI Arguments

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

### Benchmark Results

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
| Execution time | 15.62s |
| Throughput (entries/sec) | 0.32 |

**Large scale (10,000 entries, repeat-factor=2000):**

| Metric | Value |
|--------|-------|
| Input entries | 10,000 |
| Output entries | 10,000 |
| Builder windows | 362,000 |
| Filtered windows | 50,000 |
| Total filtered duration | 6,071,000s |
| Execution time | 94.77s |
| Throughput (entries/sec) | 105.52 |
| Throughput (windows/sec) | 3,819.96 |

The pipeline scales well with XennaExecutor auto-allocating 3 workers per stage on the available 8 CPU cores. Throughput increases significantly at scale as the executor amortizes startup overhead.

### Output Files

The benchmark produces three files in `--benchmark-results-path`:

| File | Description |
|------|-------------|
| `params.json` | All pipeline parameters for reproducibility |
| `metrics.json` | `is_success`, `time_taken_s`, `throughput_entries_per_sec`, `throughput_windows_per_sec`, window counts, durations |
| `tasks.pkl` | Pickled `AudioBatch` task objects for `TaskPerfUtils` aggregation |

## Performance Notes

- Both stages use Ray-based parallelism via XennaExecutor
- Processing is CPU-bound (no GPU required)
- Memory usage scales with manifest size
- For large manifests, consider processing in batches or using `--repeat-factor` for scale testing

## Troubleshooting

### No Windows Generated

- Check that `audio_sample_rate >= min_sample_rate`
- Verify `segments[].metrics.bandwidth >= min_bandwidth`
- Ensure sufficient consecutive segments for target duration
- Check speaker identifiers (avoid "no-speaker")

### Too Few Windows

- Reduce `min_speakers` requirement
- Increase `tolerance` for window duration
- Lower `min_bandwidth` threshold
- Increase `overlap_percentage` (more permissive)

### Memory Issues

- Process manifest in smaller batches
- Reduce number of parallel workers

## Related Documentation

- [Audio Getting Started Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/audio.html)
- [NeMo Curator Installation](https://docs.nvidia.com/nemo/curator/latest/get-started/installation.html)
- [Pipeline Architecture](https://docs.nvidia.com/nemo/curator/latest/about/concepts/index.html)
