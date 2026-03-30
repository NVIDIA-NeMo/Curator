# Audio Tagging Pipeline

This tutorial demonstrates how to process raw, unlabelled audio into labelled training data using NeMo Curator's audio tagging stages.

## Overview

The audio tagging pipeline is a processing framework that takes raw audio files and produces segmented, annotated manifests. It covers resampling, speaker diarization, ASR forced alignment, and merge stages.

### Pipeline Flow

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Raw Audio   │──▶│  Resample    │──▶│  Diarize     │──▶│  ASR Align   │──▶│  Merge       │
│  Manifest    │   │  (16kHz WAV) │   │  (PyAnnote)  │   │  (NeMo)      │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                                                                    │
                                                                                    ▼
                                                                          ┌──────────────────┐
                                                                          │  Output Manifest │
                                                                          └──────────────────┘
```

### Pipeline Stages

| # | Stage | Description | GPU |
|---|-------|-------------|-----|
| 0 | **ManifestReader** | Reads input JSONL manifest | No |
| 1 | **ResampleAudioStage** | Resample to 16 kHz mono WAV | No |
| 2 | **PyAnnoteDiarizationStage** | Speaker diarization and overlap detection | Yes |
| 3 | **SplitLongAudioStage** | Split segments exceeding max length | No |
| 4 | **NeMoASRAlignerStage** | Forced alignment via NeMo FastConformer | Yes |
| 5 | **JoinSplitAudioMetadataStage** | Rejoin split audio metadata | No |
| 6 | **MergeAlignmentDiarizationStage** | Merge alignment with diarization segments | No |
| 7 | **ManifestWriterStage** | Write output JSONL manifest | No |

## Installation

From the Curator repository root:

```bash
uv sync --extra audio_cuda12
source .venv/bin/activate
```

If you don't have `uv`, fall back to pip:

```bash
pip install -e ".[audio_cuda12]"
```

### Prerequisites

- **GPU**: Required for diarization (PyAnnote), ASR alignment (NeMo), and quality metrics (TorchSQUIM)
- **HuggingFace Token**: Required for PyAnnote model access. Request access at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## Quick Start

### TTS Pipeline

```bash
python tutorials/audio/tagging/main.py \
  --config-path . \
  --config-name tts_pipeline \
  input_manifest=/data/input.jsonl \
  final_manifest=/data/tts_output.jsonl \
  hf_token=<your_hf_token>
```

### ASR Pipeline

```bash
python tutorials/audio/tagging/main.py \
  --config-path . \
  --config-name asr_pipeline \
  input_manifest=/data/input.jsonl \
  final_manifest=/data/asr_output.jsonl \
  hf_token=<your_hf_token>
```

## Input Format

The input manifest should be a JSONL file where each line contains:

```json
{
  "audio_filepath": "/path/to/raw/audio.wav",
  "audio_item_id": "unique_id_001"
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio_filepath` | string | Path to the raw audio file |
| `audio_item_id` | string | Unique identifier for the audio entry |

## Output Format

The output manifest is a JSONL file where each line contains the fully processed entry:

```json
{
  "audio_filepath": "/path/to/audio.wav",
  "audio_item_id": "unique_id_001",
  "resampled_audio_filepath": "/tmp/tagging_workspace/audio_resampled/unique_id_001.wav",
  "duration": 87.13,
  "segments": [
    {
      "speaker": "unique_id_001_SPEAKER_00",
      "start": 1.23,
      "end": 6.78,
      "text": "Hello, how are you today?",
      "words": [
        {"word": "Hello", "start": 1.23, "end": 1.55},
        {"word": "how", "start": 1.60, "end": 1.72}
      ],
      "metrics": {
        "bandwidth": 8000,
        "pesq": 3.2,
        "stoi": 0.92,
        "si_sdr": 15.1
      }
    }
  ],
  "overlap_segments": []
}
```

### Output Fields

| Field | Description |
|-------|-------------|
| `resampled_audio_filepath` | Path to the resampled 16 kHz mono WAV |
| `duration` | Total audio duration in seconds |
| `segments` | List of labelled speaker segments with text, word timestamps, and quality metrics |
| `overlap_segments` | Speaker turns with detected overlap (excluded from `segments`) |

## Configuration

All parameters are defined in the YAML config files. Override from the command line:

```bash
python tutorials/audio/tagging/main.py \
  --config-path . \
  --config-name tts_pipeline \
  input_manifest=/data/input.jsonl \
  final_manifest=/data/output.jsonl \
  hf_token=<your_hf_token> \
  device=cpu \
  language_short=de \
  max_segment_length=30
```

### Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_manifest` | Path to input JSONL manifest | **Required** |
| `final_manifest` | Path for output JSONL manifest | **Required** |
| `hf_token` | HuggingFace token for PyAnnote access | `""` |
| `device` | Compute device (`cuda` or `cpu`) | `cuda` |
| `sample_rate` | Target sample rate in Hz | `16000` |
| `max_segment_length` | Maximum segment duration in seconds | `40` |
| `workspace_dir` | Directory for intermediate files | `/tmp/tagging_workspace` |
| `resampled_audio_dir` | Directory for resampled audio | `${workspace_dir}/audio_resampled` |
| `resources.cpus` | CPUs per CPU-bound stage | `2` |

### Stage-Specific Overrides

Override individual stage parameters using their index in the `stages` list:

```bash
# Change diarization model
stages.2.diarization_model=pyannote/speaker-diarization-3.1

# Adjust ASR batch size
stages.4.batch_size=16
```

## File Structure

```
tutorials/audio/tagging/
├── main.py              # Pipeline runner (YAML-driven)
├── tts_pipeline.yaml    # TTS pipeline configuration
├── asr_pipeline.yaml    # ASR pipeline configuration
└── README.md            # This file
```

## Testing

The audio tagging stages have comprehensive unit tests:

```bash
pytest tests/stages/audio/tagging/ -v
```

### Test Structure

```
tests/stages/audio/tagging/
├── conftest.py
├── test_merge_alignment_diarization.py
├── test_resample_audio.py
├── test_split.py
├── test_utils.py
└── inference/
    ├── test_base_asr_processor.py
    └── test_nemo_asr_align.py
```

## Troubleshooting

### No Segments Produced

- Ensure `hf_token` is set and has access to the PyAnnote model
- Verify input audio files exist at the paths in the manifest
- Check that `audio_item_id` is unique per entry

### GPU Out of Memory

- Reduce `stages.4.batch_size` (ASR alignment)
- Reduce `stages.2.segmentation_batch_size` (diarization)
- Process fewer files per manifest

### Slow Processing

- Ensure `device=cuda` for GPU-accelerated stages
- Increase `resources.cpus` for CPU-bound stages
- Split large manifests and process in parallel

## Related Documentation

- [Audio Getting Started Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/audio.html)
- [ALM Data Pipeline Tutorial](../alm/)
- [FLEURS Dataset Tutorial](../fleurs/)
- [NeMo Curator Installation](https://docs.nvidia.com/nemo/curator/latest/get-started/installation.html)
