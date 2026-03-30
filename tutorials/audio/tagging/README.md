# Audio Tagging Pipeline

This tutorial demonstrates how to process raw, unlabelled audio into labelled training data using NeMo Curator's audio tagging stages.

## Overview

The audio tagging pipeline is a **modality-agnostic** processing framework that takes raw audio files and produces segmented, annotated manifests. Stages 0-9 form a generic core that applies to any downstream modality -- resampling, speaker diarization, ASR forced alignment, text normalization, and quality metrics. The final **PrepareModuleSegmentsStage** (stage 10) is the only modality-specific stage: it reshapes the labelled segments into training-ready data for a target module such as **TTS** or **ASR**.

### Generic Core + Modality-Specific Output

```
                          ┌─────────────────────────────────────────────┐
                          │       Generic Tagging Core (stages 0–9)     │
                          │  Applicable to ASR, TTS, ALM, or any        │
                          │  audio modality requiring labelled data     │
                          └─────────────────────────────────────────────┘

┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Raw Audio   │──▶│  Resample    │──▶│  Diarize     │──▶│  ASR Align   │──▶│  Merge &     │
│  Manifest    │   │  (16kHz WAV) │   │  (PyAnnote)  │   │  (NeMo)      │   │  Normalize   │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                                                                    │
                                                                                    ▼
                                                                          ┌──────────────────┐
                                                                          │  Quality Metrics │
                                                                          │  (Bandwidth,     │
                                                                          │   PESQ, STOI,    │
                                                                          │   SI-SDR)        │
                                                                          └──────────────────┘
                                                                                    │
                          ┌─────────────────────────────────────────────┐           │
                          │  Modality-Specific Stage (stage 10)         │           │
                          │  PrepareModuleSegmentsStage                 │◀──────────┘
                          │                                             │
                          │  module=tts  ──▶  TTS training segments     │
                          │  module=asr  ──▶  ASR training segments     │
                          └─────────────────────────────────────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │  Output Manifest │
                                          └──────────────────┘
```

### PrepareModuleSegmentsStage and `full_utterance_ratio`

The **PrepareModuleSegmentsStage** converts the generic labelled segments into
training-ready data for the chosen modality via the `module` parameter (`tts` or `asr`).

A key parameter is **`full_utterance_ratio`**, which controls what fraction of
the output segments must be **complete sentences** -- i.e., segments whose text
ends with a terminal punctuation mark (`.`, `!`, `?`, etc. as defined by
`terminal_punct_marks`).

| Value | Meaning | Use Case |
|-------|---------|----------|
| `1.0` | **100%** of segments must end with terminal punctuation | **TTS** -- synthesisers need grammatically complete utterances for natural prosody |
| `0.8` | **80%** must end with terminal punctuation; 20% may be partial | **ASR** -- partial utterances add diversity and help the model handle real-world speech |
| `0.0` | No constraint on sentence completeness | Exploratory / other modalities |

Two ready-made configurations are provided:

| Config | Modality | `module` | `full_utterance_ratio` |
|--------|----------|----------|------------------------|
| `tts_pipeline.yaml` | Text-to-Speech | `tts` | `1.0` |
| `asr_pipeline.yaml` | Speech Recognition | `asr` | `0.8` |

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
| 7 | **InverseTextNormalizationStage** | Inverse text normalization (numbers, dates, etc.) | No |
| 8 | **BandwidthEstimationStage** | Estimate audio bandwidth per segment | No |
| 9 | **TorchSquimQualityMetricsStage** | Audio quality metrics (PESQ, STOI, SI-SDR) | Yes |
| 10 | **PrepareModuleSegmentsStage** | Prepare final segments for TTS or ASR | No |
| 11 | **ManifestWriterStage** | Write output JSONL manifest | No |

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
| `language_short` | 2-letter language code for ITN | `en` |
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

# Change segment preparation for TTS
stages.10.min_duration=3
stages.10.max_duration=15
stages.10.terminal_punct_marks=".!?"
```

### TTS vs ASR Differences

The only stage that differs between modalities is **PrepareModuleSegmentsStage** (stage 10):

| Parameter | TTS | ASR | Effect |
|-----------|-----|-----|--------|
| `stages.10.module` | `tts` | `asr` | Modality-specific segment shaping |
| `stages.10.full_utterance_ratio` | `1.0` | `0.8` | Fraction of segments that must end with terminal punctuation (`.!?`) |

See [PrepareModuleSegmentsStage and `full_utterance_ratio`](#preparemoduleSegmentsstage-and-full_utterance_ratio) above for a detailed explanation of this parameter.

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
├── test_base_asr_processor.py
├── test_datasets.py
├── test_itn.py
├── test_merge_alignment_diarization.py
├── test_metrics.py
├── test_prepare_module_segments.py
├── test_pyannote.py
├── test_split.py
├── test_text.py
├── test_utils.py
└── test_vad.py
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
