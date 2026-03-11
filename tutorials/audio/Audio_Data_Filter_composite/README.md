# Audio Data Filtration Pipeline

A CompositeStage pipeline for audio curation that extracts clean single-speaker segments with timestamp mapping back to original files.

## Overview

`AudioDataFilterStage` is a **CompositeStage** that decomposes into independent pipeline stages. The executor schedules each stage separately, enabling cross-file GPU pipelining and per-stage resource allocation.

| Feature | Description |
|---------|-------------|
| **Decomposed Pipeline** | 12 independent stages (when fully enabled) scheduled by the executor |
| **Cross-file Parallelism** | Different files can be in different stages simultaneously |
| **Timestamp Mapping** | All segments maintain original file positions via segment mappings |
| **Multi-Format Support** | WAV, MP3, FLAC, OGG, M4A, AAC, WMA, OPUS, WebM |
| **Quality Filters** | NISQA, SIGMOS, Band (bandwidth classification) |
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
AudioDataFilterStage (CompositeStage) decomposes into:

  MonoConversion (1:1)
    -> VAD batch mode (1:1, 1 item -> N segment items)
    -> BandFilter (1:1, filter items)
    -> NISQA (1:1, filter items)
    -> SIGMOS (1:1, filter items)
    -> SegmentConcatenation (1:1, M items -> 1 item + timestamp mappings)
    -> SpeakerSeparation (1:N fan-out, 1 task per speaker)
    -> VAD per speaker (1:1)
    -> BandFilter per speaker (1:1)
    -> NISQA per speaker (1:1)
    -> SIGMOS per speaker (1:1)
    -> TimestampMapper (1:1, resolve to original file positions)

Task granularity:
  - One task = one file (or one speaker after fan-out)
  - Items within a task = segments from that file
  - Cross-file parallelism via the executor
```

## Data Contract

Canonical item format between stages: `waveform` (torch.Tensor) + `sample_rate` (int).
No `item['audio']` (PyDub) in the inter-stage contract.

### Final output (after TimestampMapper):

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

## Command-Line Arguments (pipeline.py)

### Required

| Argument | Description |
|----------|-------------|
| `--raw_data_dir` | Input directory containing audio files |

### Optional

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `{raw_data_dir}/result` | Output directory |
| `--gpus` | `1.0` | GPU allocation per stage |
| `--cpus` | `1.0` | CPU cores per stage |
| `--sample_rate` | `48000` | Expected sample rate |
| `--recursive` | `false` | Search recursively |
| `--clean` | `false` | Clean output dir |
| `--verbose` | `false` | DEBUG logging |

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

## Usage from Code

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import AudioDataFilterStage, AudioDataFilterConfig
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

config = AudioDataFilterConfig(
    enable_vad=True,
    enable_nisqa=True,
    nisqa_mos_threshold=4.5,
    enable_speaker_separation=True,
)

# CompositeStage -- decomposes into 12 independent stages
pipeline = Pipeline(name="audio_curation")
pipeline.add_stage(AudioDataFilterStage(
    config=config,
    gpu_resources=Resources(gpus=1.0),
))

tasks = [AudioBatch(data={"audio_filepath": "/path/to/audio.wav"},
                    task_id="audio_001", dataset_name="my_dataset")]

from nemo_curator.backends.xenna import XennaExecutor
results = pipeline.run(XennaExecutor(), initial_tasks=tasks)
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

## Files

| File | Description |
|------|-------------|
| `audio_data_filter.py` | `AudioDataFilterStage` CompositeStage (decomposes into pipeline stages) |
| `config.py` | `AudioDataFilterConfig` dataclass |
| `pipeline.py` | Standalone Python script with CLI arguments |
| `run.py` | Hydra-based runner that loads YAML config |
| `pipeline.yaml` | YAML configuration for `run.py` |
| `extract_segments.py` | Extract audio segments from manifest |

## License

Apache License 2.0
