# Audio Data Filtration Pipeline

This module provides the `AudioDataFilterStage` - a composite pipeline stage for audio curation with consistent timestamp mapping.

## Overview

The pipeline uses `AudioDataFilterStage` which maintains **consistent timestamp mapping** through all transformations. Every output segment contains accurate `original_start_ms` and `original_end_ms` values pointing back to the source audio file.

| Feature | Description |
|---------|-------------|
| **Timestamp Mapping** | All segments maintain original file positions via `TimestampTracker` |
| **VAD Segmentation** | Voice Activity Detection to extract speech segments |
| **NISQA Filter** | Non-Intrusive Speech Quality Assessment filtering |
| **SIGMOS Filter** | Signal MOS quality filtering |
| **Band Filter** | Bandwidth classification (full-band vs narrow-band) |
| **Speaker Separation** | NeMo-based speaker diarization |

## Timestamp Mapping

The `AudioDataFilterStage` uses `TimestampTracker` to ensure consistent timestamp mapping:

```
Original Audio File
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  VAD Segmentation → Quality Filters → Concatenation             │
│       ↓                                      ↓                   │
│  Speaker Separation → Per-Speaker Filtering                     │
│       ↓                                                          │
│  TimestampTracker.translate_to_original()                        │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
Output Segments with:
  - original_file: "/path/to/source.wav"
  - original_start_ms: 1500   ← Accurate position in source
  - original_end_ms: 5200     ← Accurate position in source
  - duration_ms: 3700
  - speaker_id: "speaker_0"
  - nisqa_mos: 4.7
```

## Quick Start

### Environment Setup

```bash
# Required: Set RAY_TMPDIR to avoid socket path length issues
export RAY_TMPDIR=/tmp
```

### Option 1: Hydra + YAML (Recommended)

```bash
cd /path/to/Curator/nemo_curator/stages/audio/advance_pipelines/Audio_data_filter

python run.py \
    --config-path . \
    --config-name pipeline.yaml \
    raw_data_dir=/path/to/audio/files
```

### Option 2: Direct Python Script

```bash
cd /path/to/Curator

python nemo_curator/stages/audio/advance_pipelines/Audio_data_filter/pipeline.py \
    --raw_data_dir /path/to/audio/files \
    --enable-vad \
    --enable-nisqa \
    --enable-sigmos \
    --enable-band-filter \
    --enable-speaker-separation \
    --gpus 1.0 \
    --verbose
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         run.py                                   │
│  1. Load audio files → Create AudioBatch tasks                  │
│  2. Create Pipeline from pipeline.yaml                          │
│  3. Execute with XennaExecutor                                  │
│  4. Save results to JSONL (handles AudioBatch format)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AudioDataFilterStage                          │
│  - Mono conversion                                               │
│  - VAD segmentation                                             │
│  - Quality filtering (NISQA, SIGMOS, Band)                      │
│  - Speaker separation                                           │
│  - Timestamp mapping via TimestampTracker                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output: manifest.jsonl                        │
│  Each line: {original_file, original_start_ms, original_end_ms, │
│              duration_ms, speaker_id, quality_scores...}         │
└─────────────────────────────────────────────────────────────────┘
```

## Resource Allocation

Resource allocation can be configured directly in the config, enabling automatic parallel processing:

### Config-Based Resource Allocation

```python
from nemo_curator.stages.audio import AudioDataFilterStage, AudioDataFilterConfig

# Resources are now part of the config
config = AudioDataFilterConfig(
    cpus=4.0,   # CPU cores for parallel processing
    gpus=1.0,   # GPU allocation for model inference
    enable_nisqa=True,
    nisqa_mos_threshold=4.0,
)

# Stage automatically uses resources from config
stage = AudioDataFilterStage(config=config)
```

### Parallel Processing Behavior

| Configuration | Behavior |
|---------------|----------|
| `cpus > 1`, `gpus == 0` | CPU parallel processing via ThreadPoolExecutor |
| `gpus > 0` | GPU for model inference (NISQA, SIGMOS, VAD, Speaker Sep) |
| `cpus == 1`, `gpus == 0` | Sequential processing |

### YAML Configuration

```yaml
# In pipeline.yaml
cpus: 4.0
gpus: 1.0

processors:
  - _target_: nemo_curator.stages.audio.AudioDataFilterStage
    config:
      _target_: nemo_curator.stages.audio.AudioDataFilterConfig
      cpus: ${cpus}
      gpus: ${gpus}
      enable_nisqa: true
      # ... other settings
```

## Important Notes

### AudioBatch vs DocumentBatch

The `AudioDataFilterStage` works with `AudioBatch` (not `DocumentBatch`). This means:

- **JsonlWriter from text module does NOT work** with audio pipelines
- Results are saved by `run.py` using custom JSONL writer
- `AudioBatch.data` is a `list[dict]`, not a pandas DataFrame

### Ray Socket Path

If you see this error:
```
AF_UNIX path length cannot exceed 107 bytes
```

Set `RAY_TMPDIR` to a shorter path:
```bash
export RAY_TMPDIR=/tmp
```

## Command-Line Arguments (pipeline.py)

### Required

| Argument | Description |
|----------|-------------|
| `--raw_data_dir` | Input directory containing .wav audio files |

### Optional

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `{raw_data_dir}/result` | Output directory for manifest |
| `--gpus` | `1.0` | GPU allocation (now set via config) |
| `--cpus` | `1.0` | CPU cores for parallel processing (now set via config) |
| `--batch_size` | `1` | Batch size for processing |
| `--sample_rate` | `48000` | Expected audio sample rate |
| `--recursive` | `false` | Search input directory recursively |
| `--clean` | `false` | Clean output directory before processing |
| `--verbose` | `false` | Enable DEBUG level logging (via loguru) |

> **Note**: `--gpus` and `--cpus` are passed to `AudioDataFilterConfig` which propagates resources to all sub-stages.

### VAD Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-vad` | `false` | Enable VAD segmentation |
| `--vad-min-duration` | `2.0` | Minimum segment duration (seconds) |
| `--vad-max-duration` | `60.0` | Maximum segment duration (seconds) |
| `--vad-threshold` | `0.5` | VAD detection threshold (0-1) |

### Quality Filter Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-band-filter` | `false` | Enable bandwidth filter |
| `--band-value` | `full_band` | Band type to pass |
| `--enable-nisqa` | `false` | Enable NISQA quality filter |
| `--nisqa-mos-threshold` | `4.5` | Minimum NISQA MOS score (1-5) |
| `--nisqa-noi-threshold` | `4.3` | Minimum NISQA noisiness score (1-5) |
| `--enable-sigmos` | `false` | Enable SIGMOS quality filter |
| `--sigmos-noise-threshold` | `4.0` | Minimum SIGMOS noise score (1-5) |
| `--sigmos-ovrl-threshold` | `3.5` | Minimum SIGMOS overall score (1-5) |

### Speaker Separation Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-speaker-separation` | `false` | Enable speaker separation |
| `--speaker-exclude-overlaps` | `true` | Exclude overlapping speech |
| `--speaker-min-duration` | `0.8` | Minimum speaker segment (seconds) |

## YAML Configuration (pipeline.yaml)

Key settings in `pipeline.yaml`:

```yaml
# Input/Output
raw_data_dir: ???  # Required - set via command line
output_dir: ${raw_data_dir}/result

# Audio processing
sample_rate: 48000

# VAD
enable_vad: true
vad_min_duration_sec: 2.0
vad_max_duration_sec: 60.0

# Quality filters
enable_nisqa: true
nisqa_mos_threshold: 4.5

enable_sigmos: true
sigmos_noise_threshold: 4.0

enable_band_filter: true
band_value: full_band

# Speaker separation
enable_speaker_separation: true
speaker_min_duration: 0.8
```

Override via command line:
```bash
python run.py --config-path . --config-name pipeline.yaml \
    raw_data_dir=/path/to/audio \
    enable_nisqa=false \
    enable_speaker_separation=false
```

## Output Format

Results are written as JSONL to `{output_dir}/manifest.jsonl`:

```json
{
    "original_file": "/path/to/source_audio.wav",
    "original_start_ms": 1500,
    "original_end_ms": 5200,
    "duration_ms": 3700,
    "duration_sec": 3.7,
    "speaker_id": "speaker_0",
    "num_speakers": 2,
    "nisqa_mos": 4.7,
    "nisqa_noi": 4.5,
    "sigmos_noise": 4.2,
    "sigmos_ovrl": 3.8,
    "band_prediction": "full_band"
}
```

**Timestamp Fields:**
- `original_file`: Path to the source audio file
- `original_start_ms`: Start position in source file (milliseconds)
- `original_end_ms`: End position in source file (milliseconds)
- `duration_ms`: Segment duration in milliseconds
- `duration_sec`: Segment duration in seconds

## How Timestamp Mapping Works

The `AudioDataFilterStage` maintains timestamps through:

1. **VAD Segmentation**: Records `start_ms` and `end_ms` for each speech segment
2. **Quality Filtering**: Preserves timestamp information through filters
3. **Concatenation**: `TimestampTracker.build_from_segments()` builds a mapping table
4. **Speaker Separation**: Processes concatenated audio by speaker
5. **Timestamp Translation**: `TimestampTracker.translate_to_original()` maps final segments back to source positions

This ensures that even after complex transformations (concatenation + speaker separation), you can accurately locate each segment in the original audio file.

## Files

| File | Description |
|------|-------------|
| `audio_data_filter.py` | Main `AudioDataFilterStage` implementation |
| `config.py` | `AudioDataFilterConfig` dataclass |
| `pipeline.py` | Standalone Python script with all processing logic |
| `run.py` | Hydra-based runner that loads YAML config and saves results |
| `pipeline.yaml` | YAML configuration for `run.py` |
| `README.md` | This documentation |

## Pipeline Stages (Internal)

The `AudioDataFilterStage` internally orchestrates:

1. **MonoConversion**: Converts to mono, verifies sample rate
2. **VADSegmentation**: Extracts speech segments with timestamps
3. **BandFilter**: Filters by bandwidth classification
4. **NISQAFilter**: Filters by NISQA speech quality
5. **SIGMOSFilter**: Filters by SIGMOS signal quality
6. **SpeakerSeparation**: Separates by speaker with NeMo diarization
7. **TimestampMapping**: Maps all segments back to original file positions

## Usage from Code

```python
from nemo_curator.stages.audio import AudioDataFilterStage, AudioDataFilterConfig
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

# Create config
config = AudioDataFilterConfig(
    enable_vad=True,
    enable_nisqa=True,
    nisqa_mos_threshold=4.5,
    enable_speaker_separation=True,
)

# Create stage with GPU resources
stage = AudioDataFilterStage(config=config).with_(
    resources=Resources(gpus=1.0)
)

# Process an audio batch
task = AudioBatch(
    data={"audio_filepath": "/path/to/audio.wav"},
    task_id="audio_001",
    dataset_name="my_dataset"
)

# Setup, process, teardown
stage.setup()
result = stage.process(task)
stage.teardown()

# Result contains segments with timestamp mapping
for segment in result.data:
    print(f"Original: {segment['original_start_ms']}-{segment['original_end_ms']}ms")
```

## Configuration Parameters Reference

The `AudioDataFilterConfig` dataclass provides all configuration options for the pipeline. Below is a complete reference of all parameters:

### Resource Allocation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cpus` | `float` | `1.0` | Number of CPU cores for parallel processing |
| `gpus` | `float` | `1.0` | GPU allocation (0.0–1.0 for fractional, >1 for multiple GPUs) |

### General Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_rate` | `int` | `48000` | Target sample rate in Hz |
| `strict_sample_rate` | `bool` | `True` | Reject audio if sample rate does not match target |

### VAD (Voice Activity Detection)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_vad` | `bool` | `True` | Enable VAD segmentation |
| `vad_min_duration_sec` | `float` | `2.0` | Minimum segment duration (seconds) |
| `vad_max_duration_sec` | `float` | `60.0` | Maximum segment duration (seconds) |
| `vad_min_interval_ms` | `int` | `500` | Minimum interval between segments (ms) to avoid merging close segments |
| `vad_threshold` | `float` | `0.5` | Probability threshold for speech detection (0–1). Higher → stricter, lower → more sensitive |
| `vad_speech_pad_ms` | `int` | `300` | Padding (ms) added before and after detected speech to prevent clipping |

### Concatenation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `silence_duration_ms` | `int` | `500` | Silence duration inserted between concatenated segments (ms) |

### Band Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_band_filter` | `bool` | `True` | Enable band filtering |
| `band_value` | `Literal["full_band", "narrow_band"]` | `"full_band"` | Band to pass: `"full_band"` or `"narrow_band"` |

### NISQA Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_nisqa` | `bool` | `True` | Enable NISQA quality filter |
| `nisqa_mos_threshold` | `Optional[float]` | `4.5` | Minimum MOS score (1–5) |
| `nisqa_noi_threshold` | `Optional[float]` | `4.3` | Minimum noise quality (1–5) |
| `nisqa_col_threshold` | `Optional[float]` | `None` | Minimum coloration quality (1–5) |
| `nisqa_dis_threshold` | `Optional[float]` | `None` | Minimum discontinuity quality (1–5) |
| `nisqa_loud_threshold` | `Optional[float]` | `None` | Minimum loudness quality (1–5) |

### SIGMOS Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_sigmos` | `bool` | `True` | Enable SIGMOS quality filter |
| `sigmos_noise_threshold` | `Optional[float]` | `4.0` | Minimum noise quality (1–5) |
| `sigmos_ovrl_threshold` | `Optional[float]` | `3.5` | Minimum overall quality (1–5) |
| `sigmos_sig_threshold` | `Optional[float]` | `None` | Minimum signal quality (1–5) |
| `sigmos_col_threshold` | `Optional[float]` | `None` | Minimum coloration quality (1–5) |
| `sigmos_disc_threshold` | `Optional[float]` | `None` | Minimum discontinuity quality (1–5) |
| `sigmos_loud_threshold` | `Optional[float]` | `None` | Minimum loudness quality (1–5) |
| `sigmos_reverb_threshold` | `Optional[float]` | `None` | Minimum reverb quality (1–5) |

### Speaker Separation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_speaker_separation` | `bool` | `True` | Enable speaker diarization |
| `speaker_exclude_overlaps` | `bool` | `True` | Exclude overlapping speech segments |
| `speaker_min_duration` | `float` | `0.8` | Minimum speaker segment duration (seconds) |
| `speaker_gap_threshold` | `float` | `0.1` | Gap threshold for merging segments (seconds) |
| `speaker_buffer_time` | `float` | `0.5` | Buffer time around segments (seconds) |

## Extracting Audio Segments

After running the audio data filter pipeline, you can extract the filtered segments as individual audio files using the `extract_segments.py` script.

### Overview

The extraction script:
- Reads the `manifest.jsonl` output from the pipeline
- Extracts each segment from its original audio file using the timestamp information
- Saves segments as individual `.wav` files with structured naming
- Groups segments by speaker (if speaker separation was enabled)
- Generates an extraction summary with statistics

### Usage

```bash
python extract_segments.py \
    --manifest /path/to/result/manifest.jsonl \
    --output-dir /path/to/extracted_segments \
    --verbose
```

### Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--manifest` | `-m` | Yes | Path to manifest.jsonl file from pipeline output |
| `--output-dir` | `-o` | Yes | Directory to save extracted audio segments |
| `--verbose` | `-v` | No | Enable verbose logging with DEBUG level |

### Output File Naming Convention

Extracted segments are named using the pattern:

```
{original_filename}_speaker_{X}_segment_{YYY}.wav
```

Where:
- `{original_filename}`: Original audio file name (without extension)
- `{X}`: Speaker ID number (0, 1, 2, etc.)
- `{YYY}`: Zero-padded segment number (000, 001, 002, etc.)

**Example output files:**
```
audio_001_speaker_0_segment_000.wav
audio_001_speaker_0_segment_001.wav
audio_001_speaker_1_segment_000.wav
audio_001_speaker_1_segment_001.wav
```

### Output Structure

The output directory contains:

```
extracted_segments/
├── audio_001_speaker_0_segment_000.wav
├── audio_001_speaker_0_segment_001.wav
├── audio_001_speaker_1_segment_000.wav
├── audio_002_speaker_0_segment_000.wav
├── ...
└── extraction_summary.json
```

### Extraction Summary

The script generates `extraction_summary.json` with statistics:

```json
{
  "manifest_path": "/path/to/manifest.jsonl",
  "output_dir": "/path/to/extracted_segments",
  "total_segments": 234,
  "total_duration_sec": 1456.78,
  "segments_by_speaker": {
    "speaker_0": 120,
    "speaker_1": 114
  },
  "original_files_processed": 5
}
```

### Complete Example Workflow

```bash
# Step 1: Run the audio data filter pipeline
python run.py \
    --config-path . \
    --config-name pipeline.yaml \
    raw_data_dir=/data/audio_files

# Step 2: Extract filtered segments
python extract_segments.py \
    --manifest /data/audio_files/result/manifest.jsonl \
    --output-dir /data/extracted_segments \
    --verbose

# Output:
# ============================================================
# EXTRACTION COMPLETE
# ============================================================
# Total segments extracted: 234
# Total duration: 1456.78s (24.3 min)
# Output directory: /data/extracted_segments
#
# Segments by speaker:
#   speaker_0: 120 segments
#   speaker_1: 114 segments
#
# Summary saved to: /data/extracted_segments/extraction_summary.json
```

### Features

- **Timestamp Accuracy**: Uses the precise `original_start_ms` and `original_end_ms` from the manifest
- **Automatic Grouping**: Organizes segments by speaker when speaker separation is enabled
- **Sequential Numbering**: Segments are numbered sequentially within each speaker
- **Error Handling**: Skips missing files and logs errors without stopping
- **Summary Statistics**: Provides detailed extraction statistics in JSON format

### Requirements

The extraction script requires:
- `pydub` for audio manipulation
- `loguru` for logging
- Original audio files must be accessible at the paths specified in `original_file` field

### Troubleshooting

**Issue: "Original file not found"**
- Ensure the original audio files are still at their original paths
- The `original_file` field in manifest.jsonl contains absolute paths

**Issue: Import errors**
- Install required dependencies: `pip install pydub loguru`
- For pydub, you may also need ffmpeg: `apt-get install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS)

**Issue: Out of memory**
- Process files in batches if you have many large audio files
- The script loads each original file once and extracts all segments from it

## License

Apache License 2.0

