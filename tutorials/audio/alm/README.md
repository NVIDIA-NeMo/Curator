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

Install NeMo Curator and tutorial dependencies:

```bash
# From the Curator repository root
pip install -e .

# Install additional tutorial dependencies
pip install -r tutorials/audio/alm/requirements.txt
```

## Sample Data

Sample data is located in `tests/fixtures/audio/alm/` for use in both testing and tutorials:

```
tests/fixtures/audio/alm/
└── sample_input.jsonl        # 5 sample audio manifests with diarized segments

tutorials/audio/alm/
├── main.py                   # Pipeline runner (YAML-driven)
├── pipeline.yaml             # Pipeline configuration
├── requirements.txt          # Additional dependencies
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
  input_manifest=tests/fixtures/audio/alm/sample_input.jsonl
```

Expected output:
```
PIPELINE COMPLETE
==================================================
  Input entries: 5
  Output entries: 5
  Entries with windows: 5
  Stage 1 (Builder) windows: 181
  Stage 2 (Overlap) windows: 25
  Total filtered duration: 3035.50s (50.59 min)
Results saved to: ./alm_output/alm_output.jsonl
```

## Using Custom Data

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  input_manifest=/path/to/your/data.jsonl \
  output_dir=./my_output
```

## Configuration

All parameters are defined in `pipeline.yaml`. Override from command line:

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  input_manifest=/data/input.jsonl \
  output_dir=./custom_output \
  processors.0.min_speakers=3 \
  processors.0.max_speakers=6 \
  processors.1.overlap_percentage=30
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_manifest` | Path to input JSONL manifest | Required |
| `output_dir` | Directory for output files | `./alm_output` |
| `processors.0.target_window_duration` | Target window duration (seconds) | `120.0` |
| `processors.0.tolerance` | Duration tolerance (e.g., 0.1 = ±10%) | `0.1` |
| `processors.0.min_sample_rate` | Minimum sample rate (Hz) | `16000` |
| `processors.0.min_bandwidth` | Minimum bandwidth (Hz) | `8000` |
| `processors.0.min_speakers` | Minimum speakers per window | `2` |
| `processors.0.max_speakers` | Maximum speakers per window | `5` |
| `processors.1.overlap_percentage` | Overlap threshold 0-100 | `50` |

### Override Notes

Match indices in `processors` list in `pipeline.yaml`:
- `processors.0.*`: ALMDataBuilderStage parameters
- `processors.1.*`: ALMDataOverlapStage parameters

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
  input_manifest=tests/fixtures/audio/alm/sample_input.jsonl \
  processors.0.target_window_duration=60 \
  processors.0.tolerance=0.15
```

### Stricter Speaker Requirements

For exactly 2-3 speakers:

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  input_manifest=tests/fixtures/audio/alm/sample_input.jsonl \
  processors.0.min_speakers=2 \
  processors.0.max_speakers=3
```

### Aggressive Overlap Filtering

Remove all overlapping windows:

```bash
python tutorials/audio/alm/main.py \
  --config-path . \
  --config-name pipeline \
  input_manifest=tests/fixtures/audio/alm/sample_input.jsonl \
  processors.1.overlap_percentage=0
```

## Performance Notes

- Both stages use Ray-based parallelism via XennaExecutor
- Processing is CPU-bound (no GPU required)
- Memory usage scales with manifest size
- For large manifests, consider processing in batches

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
