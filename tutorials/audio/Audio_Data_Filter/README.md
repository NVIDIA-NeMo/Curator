# Audio Data Filtration Pipeline

A composite pipeline stage for audio curation with consistent timestamp mapping.

## Overview

The `AudioDataFilterStage` maintains **consistent timestamp mapping** through all transformations. Every output segment contains accurate `original_start_ms` and `original_end_ms` values pointing back to the source audio file.

| Feature | Description |
|---------|-------------|
| **Multi-Format Support** | WAV, MP3, FLAC, OGG, M4A, AAC, WMA, OPUS, WebM |
| **Timestamp Mapping** | All segments maintain original file positions via `TimestampTracker` |
| **VAD Segmentation** | Voice Activity Detection to extract speech segments |
| **NISQA Filter** | Non-Intrusive Speech Quality Assessment filtering |
| **SIGMOS Filter** | Signal MOS quality filtering |
| **Band Filter** | Bandwidth classification (full-band vs narrow-band) |
| **Speaker Separation** | NeMo-based speaker diarization |


## Quick Start

```bash
export RAY_TMPDIR=/tmp

# Option 1: Hydra + YAML (Recommended)
python run.py \
    --config-path . \
    --config-name pipeline.yaml \
    raw_data_dir=/path/to/audio/files

# Option 2: Direct Python Script
python pipeline.py \
    --raw_data_dir /path/to/audio/files \
    --enable-vad \
    --enable-nisqa \
    --enable-sigmos \
    --enable-band-filter \
    --enable-speaker-separation \
    --gpus 1.0
```

## Architecture

```
run.py / pipeline.py
  1. Load audio files -> Create AudioBatch tasks
  2. Execute AudioDataFilterStage
  3. Save results to manifest.jsonl

AudioDataFilterStage (internally):
  Mono Conversion -> VAD -> Band Filter -> NISQA -> SIGMOS
    -> Concatenation -> Speaker Separation -> Timestamp Mapping

Output: manifest.jsonl
  {original_file, original_start_ms, original_end_ms,
   duration_ms, speaker_id, quality_scores...}
```

## Command-Line Arguments (pipeline.py)

### Required

| Argument | Description |
|----------|-------------|
| `--raw_data_dir` | Input directory containing audio files |

### Optional

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `{raw_data_dir}/result` | Output directory |
| `--gpus` | `1.0` | GPU allocation |
| `--cpus` | `1.0` | CPU cores |
| `--batch_size` | `1` | Batch size |
| `--sample_rate` | `48000` | Expected sample rate |
| `--recursive` | `false` | Search recursively |
| `--clean` | `false` | Clean output dir |
| `--verbose` | `false` | DEBUG logging |
| `--input-formats` | all | Formats to process (e.g., `wav mp3 flac`) |
| `--output-format` | `wav` | Output format for segments |

### Filter Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-vad` | `false` | Enable VAD segmentation |
| `--vad-min-duration` | `2.0` | Min segment duration (sec) |
| `--vad-max-duration` | `60.0` | Max segment duration (sec) |
| `--vad-threshold` | `0.5` | VAD threshold (0-1) |
| `--enable-band-filter` | `false` | Enable bandwidth filter |
| `--band-value` | `full_band` | Band type to pass |
| `--enable-nisqa` | `false` | Enable NISQA filter |
| `--nisqa-mos-threshold` | `4.5` | Min NISQA MOS (1-5) |
| `--nisqa-noi-threshold` | `4.3` | Min NISQA noisiness (1-5) |
| `--enable-sigmos` | `false` | Enable SIGMOS filter |
| `--sigmos-noise-threshold` | `4.0` | Min SIGMOS noise (1-5) |
| `--sigmos-ovrl-threshold` | `3.5` | Min SIGMOS overall (1-5) |
| `--enable-speaker-separation` | `false` | Enable speaker separation |
| `--speaker-exclude-overlaps` | `true` | Exclude overlapping speech |
| `--speaker-min-duration` | `0.8` | Min speaker segment (sec) |

## AudioDataFilterConfig Reference

All parameters available in `AudioDataFilterConfig` (usable via YAML or Python code).

### Resource Allocation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cpus` | `float` | `1.0` | CPU cores for parallel processing |
| `gpus` | `float` | `1.0` | GPU allocation (0.0-1.0 fractional, >1 for multiple) |

### General

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_rate` | `int` | `48000` | Target sample rate (Hz) |
| `strict_sample_rate` | `bool` | `True` | Reject audio if sample rate does not match target |
| `input_formats` | `tuple` | all supported | Audio formats to discover (e.g., `(".wav", ".mp3")`) |
| `output_format` | `str` | `"wav"` | Output format for exported segments |

### VAD (Voice Activity Detection)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_vad` | `bool` | `True` | Enable VAD segmentation |
| `vad_min_duration_sec` | `float` | `2.0` | Minimum segment duration (seconds) |
| `vad_max_duration_sec` | `float` | `60.0` | Maximum segment duration (seconds) |
| `vad_min_interval_ms` | `int` | `500` | Minimum interval between segments (ms) to avoid merging close segments |
| `vad_threshold` | `float` | `0.5` | Speech detection threshold (0-1). Higher = stricter |
| `vad_speech_pad_ms` | `int` | `300` | Padding (ms) added before/after detected speech to prevent clipping |

### Concatenation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `silence_duration_ms` | `int` | `500` | Silence inserted between concatenated segments (ms) |

### Band Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_band_filter` | `bool` | `True` | Enable bandwidth classification |
| `band_value` | `str` | `"full_band"` | Band to pass: `"full_band"` or `"narrow_band"` |

### NISQA Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_nisqa` | `bool` | `True` | Enable NISQA quality filter |
| `nisqa_mos_threshold` | `float?` | `4.5` | Min MOS score (1-5) |
| `nisqa_noi_threshold` | `float?` | `4.3` | Min noise quality (1-5) |
| `nisqa_col_threshold` | `float?` | `None` | Min coloration quality (1-5) |
| `nisqa_dis_threshold` | `float?` | `None` | Min discontinuity quality (1-5) |
| `nisqa_loud_threshold` | `float?` | `None` | Min loudness quality (1-5) |

### SIGMOS Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_sigmos` | `bool` | `True` | Enable SIGMOS quality filter |
| `sigmos_noise_threshold` | `float?` | `4.0` | Min noise quality (1-5) |
| `sigmos_ovrl_threshold` | `float?` | `3.5` | Min overall quality (1-5) |
| `sigmos_sig_threshold` | `float?` | `None` | Min signal quality (1-5) |
| `sigmos_col_threshold` | `float?` | `None` | Min coloration quality (1-5) |
| `sigmos_disc_threshold` | `float?` | `None` | Min discontinuity quality (1-5) |
| `sigmos_loud_threshold` | `float?` | `None` | Min loudness quality (1-5) |
| `sigmos_reverb_threshold` | `float?` | `None` | Min reverb quality (1-5) |

### Speaker Separation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_speaker_separation` | `bool` | `True` | Enable speaker diarization |
| `speaker_exclude_overlaps` | `bool` | `True` | Exclude overlapping speech segments |
| `speaker_min_duration` | `float` | `0.8` | Minimum speaker segment duration (seconds) |
| `speaker_gap_threshold` | `float` | `0.1` | Gap threshold for merging adjacent segments (seconds) |
| `speaker_buffer_time` | `float` | `0.5` | Buffer time around speaker segments (seconds) |

## YAML Configuration (pipeline.yaml)

Override settings via command line:
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

## Usage from Code

```python
from nemo_curator.stages.audio import AudioDataFilterStage, AudioDataFilterConfig
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

config = AudioDataFilterConfig(
    enable_vad=True,
    enable_nisqa=True,
    nisqa_mos_threshold=4.5,
    enable_speaker_separation=True,
)

stage = AudioDataFilterStage(config=config).with_(
    resources=Resources(gpus=1.0)
)

task = AudioBatch(
    data={"audio_filepath": "/path/to/audio.wav"},
    task_id="audio_001",
    dataset_name="my_dataset"
)

stage.setup()
result = stage.process(task)
stage.teardown()

for segment in result.data:
    print(f"Original: {segment['original_start_ms']}-{segment['original_end_ms']}ms")
```

## Extracting Audio Segments

After running the pipeline, extract filtered segments as individual audio files:

```bash
python extract_segments.py \
    --manifest /path/to/result/manifest.jsonl \
    --output-dir /path/to/extracted_segments \
    --output-format wav \
    --verbose
```

Output files follow the naming pattern: `{filename}_speaker_{X}_segment_{YYY}.wav`

An `extraction_summary.json` with statistics is also generated.

## Files

| File | Description |
|------|-------------|
| `audio_data_filter.py` | Main `AudioDataFilterStage` implementation |
| `config.py` | `AudioDataFilterConfig` dataclass |
| `pipeline.py` | Standalone Python script with all processing logic |
| `run.py` | Hydra-based runner that loads YAML config |
| `pipeline.yaml` | YAML configuration for `run.py` |
| `extract_segments.py` | Extract audio segments from manifest |

## Important Notes

- The `AudioDataFilterStage` works with `AudioBatch` (not `DocumentBatch`). Results are saved by `run.py` using a custom JSONL writer.
- If you see `AF_UNIX path length cannot exceed 107 bytes`, set `export RAY_TMPDIR=/tmp`.

## License

Apache License 2.0
