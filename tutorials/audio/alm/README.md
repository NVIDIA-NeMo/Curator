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

With a single manifest file:

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  manifest_path=/path/to/your/data.jsonl \
  output_dir=./my_output
```

With a directory (recursively discovers all `.jsonl` and `.json` files in subdirectories):

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  manifest_path=/data/manifests/ \
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
| `stages.1.truncation` | Truncate segments exceeding window | `true` |
| `stages.1.drop_fields` | Comma-separated fields to drop from segments | `"words"` |
| `stages.1.drop_fields_top_level` | Comma-separated top-level fields to drop | `"words,segments"` |
| `stages.2.overlap_percentage` | Overlap threshold 0-100 | `50` |
| `stages.2.target_duration` | Target duration for overlap comparison | `120.0` |
| `stages.3.output_path` | Output JSONL path | `${output_dir}/alm_output.jsonl` |

### Override Notes

Match indices in `stages` list in `pipeline.yaml`:
- `stages.0.*`: ALMManifestReaderStage parameters
- `stages.1.*`: ALMDataBuilderStage parameters
- `stages.2.*`: ALMDataOverlapStage parameters
- `stages.3.*`: ALMManifestWriterStage parameters

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

See [benchmarking/ALM_BENCHMARK.md](../../../benchmarking/ALM_BENCHMARK.md) for the full ALM benchmark documentation, including how to run benchmarks, configuration, CLI arguments, and reference results.

## Testing

The ALM pipeline has comprehensive unit and integration tests in `tests/stages/audio/alm/`.

### Running Tests

From the Curator repository root:

```bash
pytest tests/stages/audio/alm/ -v
```

### Test Structure

```
tests/stages/audio/alm/
├── conftest.py                    # Shared fixtures
├── test_alm_manifest_reader.py    # 11 tests (2 classes)
├── test_alm_manifest_writer.py    # 12 tests (2 classes)
├── test_alm_data_builder.py       #  8 tests (2 classes)
└── test_alm_data_overlap.py       #  7 tests (2 classes)
```

### Shared Fixtures (`conftest.py`)

| Fixture | Description |
|---------|-------------|
| `sample_entries` | Loads all 5 entries from `tests/fixtures/audio/alm/sample_input.jsonl` |
| `sample_entry` | First entry from `sample_entries` |
| `entry_with_windows` | `sample_entry` processed through `ALMDataBuilderStage` (pre-built windows for overlap tests) |

### ALMManifestReaderStage Tests

**`TestALMManifestReader`** (unit tests):

| Test | What it verifies |
|------|-----------------|
| `test_reads_single_manifest` | Reads 2-entry JSONL, returns `AudioBatch` per entry |
| `test_reads_multiple_manifests` | Accepts list of manifest paths, concatenates entries |
| `test_one_audio_batch_per_entry` | Each entry becomes exactly one `AudioBatch` with `len(data) == 1` |
| `test_skips_blank_lines` | Blank/whitespace-only lines in JSONL are ignored |
| `test_empty_manifest` | Empty file returns `[]` |
| `test_preserves_nested_data` | Nested `segments[].metrics.bandwidth` survives round-trip |
| `test_duplicate_manifests_for_repeat` | Same path repeated 3x produces 3 batches (repeat-factor pattern) |
| `test_manifest_path_coerced_to_list` | Tuple input is coerced to list |
| `test_string_path_stays_string` | Single string path is not wrapped |
| `test_xenna_stage_spec` | Returns `{"num_workers_per_node": 1}` |
| `test_ray_stage_spec` | Returns `{"is_fanout_stage": True}` |

**`TestALMManifestReaderIntegration`**:

| Test | What it verifies |
|------|-----------------|
| `test_reads_sample_fixture` | Reads the real `sample_input.jsonl` fixture, verifies 5 entries with segments |

### ALMManifestWriterStage Tests

**`TestALMManifestWriter`** (unit tests):

| Test | What it verifies |
|------|-----------------|
| `test_writes_entries_to_jsonl` | 2 entries written as 2 JSONL lines with correct `audio_filepath` |
| `test_returns_file_group_task` | Returns `FileGroupTask` with output path, task_id, dataset_name |
| `test_propagates_metadata_and_stage_perf` | `_metadata` and `_stage_perf` pass through to output task |
| `test_appends_across_multiple_process_calls` | 3 sequential `process()` calls produce 3 lines |
| `test_setup_truncates_existing_file` | `setup()` clears pre-existing file content |
| `test_setup_creates_parent_directories` | `setup()` creates nested directories for output path |
| `test_handles_unicode_content` | Japanese and accented characters survive write/read |
| `test_preserves_nested_structures` | `windows[].segments[]` and `stats` dict survive serialization |
| `test_empty_data_writes_nothing` | Empty `data=[]` writes no lines, still returns `FileGroupTask` |
| `test_num_workers_returns_one` | `num_workers()` returns 1 (single-writer constraint) |
| `test_xenna_stage_spec` | Returns `{"num_workers": 1}` |

**`TestALMManifestWriterRoundTrip`**:

| Test | What it verifies |
|------|-----------------|
| `test_reader_writer_round_trip` | Write all fixture entries with writer, read back with reader, verify `audio_filepath` and segment counts match |

### ALMDataBuilderStage Tests

**`TestALMDataBuilder`** (unit tests):

| Test | What it verifies |
|------|-----------------|
| `test_creates_windows_from_sample` | Sample entry produces non-empty `windows` list and `stats` |
| `test_filters_low_sample_rate` | Entry with 8kHz sample rate has `lost_sr > 0` or empty windows |
| `test_filters_low_bandwidth` | All segments set to 4kHz bandwidth triggers `lost_bw > 0` |
| `test_speaker_constraints` | Single-speaker entry with `min_speakers=2` produces zero windows |
| `test_empty_segments` | Entry with `segments=[]` returns empty windows |
| `test_drop_fields` | `words` removed from segments inside windows; `words` and `segments` removed from top-level |
| `test_different_sample_rates` | All 5 fixture entries (16-48kHz) process without error |

**`TestALMDataBuilderIntegration`**:

| Test | What it verifies |
|------|-----------------|
| `test_processes_all_sample_entries` | All 5 fixture entries produce exactly **181 total windows** |

### ALMDataOverlapStage Tests

**`TestALMDataOverlap`** (unit tests):

| Test | What it verifies |
|------|-----------------|
| `test_filters_overlapping_windows` | `filtered_windows` count <= input `windows` count |
| `test_keeps_closer_to_target` | Aggressive filtering (`overlap_percentage=0`) produces valid output |
| `test_permissive_mode` | `overlap_percentage=100` keeps >= windows than `overlap_percentage=0` |
| `test_no_windows` | Entry with `windows=[]` passes through unchanged |
| `test_validation` | Invalid `overlap_percentage` (-1, 101) and `target_duration` (-1) raise `ValueError` |
| `test_calculates_duration` | Output includes `filtered_dur >= 0` and `filtered_dur_list` |

**`TestALMDataOverlapIntegration`**:

| Test | What it verifies |
|------|-----------------|
| `test_full_pipeline` | Full Builder -> Overlap pipeline: 5 entries produce **181 windows -> 25 filtered windows**, total filtered duration **~3035.5 seconds** |

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
