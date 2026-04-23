# Audio Tagging Pipeline

This tutorial demonstrates how to process raw, unlabelled audio into labelled training data using NeMo Curator's audio tagging stages.

## Overview

The audio tagging pipeline is a generic processing framework that takes raw audio files and produces segmented, annotated manifests suitable for training multiple speech modalities — **TTS**, **ASR**, **ALM**, and others. The core pipeline (stages 0–9) is shared across all modalities: resampling, speaker diarization, ASR forced alignment, merge, quality metrics, and segment preparation. The `PrepareModuleSegmentsStage` is the key stage where segments are shaped differently based on the target modality (e.g. duration constraints, utterance completeness). Optionally, a second-pass ASR transcription and WER computation can be appended to further validate transcript quality.

### Pipeline Flow

```
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│ Manifest   │─▶│ Resample   │─▶│ Diarize    │─▶│ Split Long │
│ Reader     │  │ (16kHz WAV)│  │ (PyAnnote) │  │ Audio      │
└────────────┘  └────────────┘  └────────────┘  └────────────┘
                                                      │
┌────────────┐  ┌────────────┐  ┌────────────┐       │
│ Merge      │◀─│ Join Split │◀─│ ASR Align  │◀──────┘
│ Align+Diar │  │ Metadata   │  │ (NeMo)     │
└────────────┘  └────────────┘  └────────────┘
      │
      ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│ Bandwidth  │─▶│ SQUIM      │─▶│ Prepare    │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
│ Estimation │  │ Metrics    │  │ Module Seg │  (tts / asr / …)
└────────────┘  └────────────┘  └────────────┘                      │
                                      │
                                      ▼                             │
                                ┌────────────┐  ┌────────────┐
                                │ ASR Align  │─▶│ Compute    │      │
                                │ (2nd pass) │  │ WER        │  optional
                                └────────────┘  └────────────┘      │
                                                      │
                                                      ▼             │
                                                ┌────────────┐
                                                │ Manifest   │◀ ─ ─ ┘
                                                │ Writer     │
                                                └────────────┘
```

The dashed path shows that `ManifestWriter` can follow directly after `PrepareModuleSegments` (e.g. the default TTS config) or after the optional second-pass ASR + WER stages (e.g. the ASR config).

### Pipeline Stages

#### Core Stages (shared by all modalities, stages 0–9)

| # | Stage | Description | GPU |
|---|-------|-------------|-----|
| 0 | **ManifestReader** | Reads input JSONL manifest | No |
| 1 | **ResampleAudioStage** | Resample to 16 kHz mono WAV | No |
| 2 | **PyAnnoteDiarizationStage** | Speaker diarization and overlap detection | Yes |
| 3 | **SplitLongAudioStage** | Split segments exceeding max length | No |
| 4 | **NeMoASRAlignerStage** | Forced alignment via NeMo FastConformer | Yes |
| 5 | **JoinSplitAudioMetadataStage** | Rejoin split audio metadata | No |
| 6 | **MergeAlignmentDiarizationStage** | Merge alignment with diarization segments | No |
| 7 | **BandwidthEstimationStage** | Spectral bandwidth estimation per segment | No |
| 8 | **TorchSquimQualityMetricsStage** | PESQ, STOI, SI-SDR quality metrics | Yes |
| 9 | **PrepareModuleSegmentsStage** | Merge/split segments for the target modality by duration, pauses, and punctuation. Controlled by the `module` parameter (`tts`, `asr`, etc.) | No |

#### Optional Second-Pass ASR & WER Stages

These stages can be appended after `PrepareModuleSegments` in any modality config to cross-validate transcripts:

| # | Stage | Description | GPU |
|---|-------|-------------|-----|
| 10 | **NeMoASRAlignerStage** (2nd pass) | Second-pass ASR transcription (e.g. CTC Conformer) | Yes |
| 11 | **ComputeWERStage** | Word/character error rate between first and second ASR transcripts | No |

#### Optional Text Normalization Stages

These stages can be inserted after merging (stage 6) for language-specific text processing:

| Stage | Description | GPU |
|-------|-------------|-----|
| **InverseTextNormalizationStage** | Inverse text normalization (spoken → written) | No |
| **ChineseConversionStage** | Traditional → Simplified Chinese conversion | No |
| **ArabicRemoveDiacriticsStage** | Remove diacritics from Arabic text | No |
| **PNCwithBERTStage** | Punctuation & capitalization via NeMo BERT (requires `nemo_toolkit <= 2.4.1`) | Yes |

## Installation

From the Curator repository root:

```bash
uv sync --extra audio_cuda12
source .venv/bin/activate
```

### Prerequisites

- **System packages**: `ffmpeg` and `sox` must be installed for audio resampling and format conversion:
  ```bash
  # Ubuntu / Debian
  sudo apt-get install -y ffmpeg sox libsox-fmt-all

  ```
- **GPU**: Required for diarization (PyAnnote), VAD (Pyannote), ASR alignment (NeMo)
- **HuggingFace Token**: Required for PyAnnote model access. Request access at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection)

## Quick Start

### TTS Pipeline

The TTS config runs the core stages with `module: tts` in `PrepareModuleSegmentsStage` (`full_utterance_ratio: 1.0`). The output segments are single-speaker utterances, each annotated with quality metrics such as `bandwidth`, `stoi_squim`, `si_sdr`, and `pesq_squim`. These metrics can be used downstream to filter for high-quality audio — for example, keeping only segments where `bandwidth >= 8000 && si_sdr >= 15 && stoi_squim >= 0.9`.

A small toy dataset is bundled in `tests/fixtures/audio/tagging/` so you can run end-to-end without providing your own audio:

```bash
python tutorials/audio/tagging/main.py \
  --config-path . \
  --config-name tts_pipeline \
  input_manifest=tests/fixtures/audio/tagging/sample_input.jsonl \
  final_manifest=/tmp/tts_output.jsonl \
  hf_token=<your_hf_token>
```

### ASR Pipeline

The ASR config runs the same core stages with `module: asr` (`full_utterance_ratio: 0.8` to allow partial utterances), then adds second-pass ASR and WER computation. The per-segment `wer` field can be used to filter for reliable transcripts — for example, keeping only segments where `wer <= 10%`.

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
        {"word": "how", "start": 1.60, "end": 1.72} ...
      ],
      "metrics":
        {
          "bandwidth": [8000, 8400, 7200, ...],
          "pesq_squim": [3.4, 3.5, 3.6, ...],
          "stoi_squim": [0.91, 0.92, 0.90, ...],
          "si_sdr": [19.8, 20.4, 21.0, ...],
        }
    }
  ],
  "overlap_segments": [],
  "text": "Hello, how are you today? Let's get started with the tutorial.",
  "alignment": [
    {"word": "Hello", "start": 1.23, "end": 1.55},
    {"word": "how", "start": 1.60, "end": 1.72}, ...
  ],
}
```

### Output Fields

| Field                     | Source                  | Description                                                          |
|---------------------------|-------------------------|----------------------------------------------------------------------|
| `resampled_audio_filepath`| Core                    | Path to the resampled 16 kHz mono WAV                                |
| `duration`                | Core                    | Total audio duration in seconds                                      |
| `segments`                | Core                    | List of labelled speaker segments with text, word timestamps         |
| `overlap_segments`        | Core                    | Speaker turns with detected overlap (excluded from `segments`)       |
| `text`                    | Core                    | Full transcript text for the audio entry                             |
| `alignment`               | Core                    | List of word-level alignment objects (`word`, `start`, `end`)        |
| `segments[].bandwidth`    | Core                    | Estimated spectral bandwidth                                         |
| `segments[].pesq_squim`   | Core                    | PESQ quality score (via TorchSQUIM)                                  |
| `segments[].stoi_squim`   | Core                    | STOI quality score (via TorchSQUIM)                                  |
| `segments[].si_sdr`       | Core                    | SI-SDR quality score (via TorchSQUIM)                                |
| `segments[].text_2`       | Optional (2nd-pass ASR) | Second-pass ASR transcript (e.g. CTC Conformer)                     |
| `segments[].wer`          | Optional (ComputeWER)   | Word error rate between first and second ASR transcripts             |

## Configuration

All parameters are defined in the YAML config files. Override from the command line:

```bash
python tutorials/audio/tagging/main.py \
  --config-path . \
  --config-name tts_pipeline \
  input_manifest=tests/fixtures/audio/tagging/sample_input.jsonl \
  final_manifest=/tmp/output.jsonl \
  hf_token=<your_hf_token> \
  language_short=de \
  max_segment_length=30
```

### Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_manifest` | Path to input JSONL manifest | **Required** |
| `final_manifest` | Path for output JSONL manifest | **Required** |
| `hf_token` | HuggingFace token for PyAnnote access | `""` |
| `sample_rate` | Target sample rate in Hz | `16000` |
| `max_segment_length` | Maximum segment duration in seconds | `40` |
| `workspace_dir` | Directory for intermediate files | `/tmp/tagging_workspace` |
| `resampled_audio_dir` | Directory for resampled audio | `${workspace_dir}/audio_resampled` |
| `resources.cpus` | CPUs per CPU-bound stage | `2` |

### Stage-Specific Overrides

Override individual stage parameters using their index in the `stages` list:

```bash
# Change diarization model (stage 2)
stages.2.diarization_model=pyannote/speaker-diarization-3.1

# Adjust first-pass ASR batch size (stage 4)
stages.4.batch_size=16

# Adjust PrepareModuleSegments duration limits (stage 9)
stages.9.min_duration=3 stages.9.max_duration=25

# Adjust second-pass ASR batch size (stage 10, when present)
stages.10.batch_size=32
```

## Parameter Tuning

### `max_segment_length` (default: 40s)

Controls the maximum duration of audio segments fed to the first pass ASR. This is the single most impactful parameter for output quality. Choose this value according to the better accuracy for the asr model.

| Value | Effect | Best for |
|-------|--------|----------|
| 20s | Shorter segments, more split points. Higher diarization accuracy but more ASR boundary errors. | Short-form content (podcasts, interviews) |
| 40s | Balanced default. Works well for most conversational audio. | General purpose |
| 60s | Fewer splits, longer context for ASR. Risk of mixed-speaker segments. | Long monologues, lectures |

### `segmentation_batch_size` (PyAnnote diarization)

Controls GPU memory vs throughput for the diarization model:

| Value | GPU Memory | Throughput |
|-------|-----------|------------|
| 32 | ~2 GB | Slower, safe for T4 (16 GB) alongside ASR |
| 128 (default) | ~6 GB | Good balance for A100 |
| 256+ | ~10+ GB | Maximum throughput, requires ≥40 GB VRAM |

### `transcribe_batch_size` (NeMo ASR Aligner, default: 32)

Controls how many audio chunks are transcribed in a single forward pass. Reduce to 8–16 if you see CUDA OOM errors during the ASR alignment stage.

## GPU Memory Requirements

The pipeline loads two GPU models simultaneously at peak:

| Model | VRAM | Stage |
|-------|------|-------|
| PyAnnote speaker diarization | ~2–3 GB | Stage 2 |
| PyAnnote segmentation | ~1–2 GB | Stage 2 |
| NeMo FastConformer (1.1B, CTC) | ~3–4 GB | Stage 4 |

**Total peak VRAM**: ~6–9 GB (models are loaded sequentially by default, not concurrently).

| GPU | Fits? | Notes |
|-----|-------|-------|
| T4 (16 GB) | Yes | Reduce `segmentation_batch_size` to 32 and `transcribe_batch_size` to 8 |
| A10G (24 GB) | Yes | Default settings work |
| A100 (40/80 GB) | Yes | Can increase batch sizes for throughput |

## Timing Estimates

Approximate wall-clock time per hour of input audio on a single A100-40GB:

| Stage | Time per hour of audio | Notes |
|-------|----------------------|-------|
| Resample | ~10s | CPU-bound, I/O limited |
| PyAnnote Diarization | ~2–4 min | GPU, depends on speaker count |
| Split + ASR Alignment | ~3–5 min | GPU, depends on segment count |
| Merge + Write | ~5s | CPU-only |
| **Total** | **~6–10 min / hr of audio** | |

> **First run is slower**: model weights (~1.3 GB total) are downloaded on the first execution. See [Troubleshooting](#first-run-appears-hung) below.

## Expected Filtering Ratios

After diarization, not all audio ends up in the final output:

| Category | Typical % of total duration | Description |
|----------|-----------------------------|-------------|
| Speaker segments | 70–85% | Clean, single-speaker audio |
| Overlap segments | 10–20% | Multi-speaker overlap, excluded from `segments` |
| No-speaker / silence | 5–15% | Gaps between speaker turns |

These ratios vary significantly by content type. Interviews (2 speakers, turn-taking) yield higher usable percentages than panel discussions (4+ speakers, frequent overlap).

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
├── test_prepare_module_segments.py
├── test_resample_audio.py
├── test_split.py
├── test_utils.py
├── inference/
│   ├── test_base_asr_processor.py
│   └── test_nemo_asr_align.py
├── metrics/
│   └── test_metrics.py
└── text/
    ├── test_itn.py
    └── test_text.py
```

### End-to-End Pipeline Test

Automated end-to-end (E2E) tests validate the full TTS and ASR audio tagging pipelines. These tests mirror the tutorial configurations and ensure all pipeline stages work together as expected.

To run the E2E tests:

```bash
pytest tests/stages/audio/tagging/e2e/ -v
```

**Relevant files:**

```
tests/stages/audio/tagging/e2e/
├── test_tts_e2e.py         # End-to-end TTS tagging pipeline test
├── test_asr_e2e.py         # End-to-end ASR tagging pipeline test
├── conftest.py             # Test fixtures (manifests, input data)
├── utils.py                # Output validation helpers
└── configs/
    ├── tts_pipeline.yaml   # Test configuration for the TTS pipeline
    └── asr_pipeline.yaml   # Test configuration for the ASR pipeline
```

> **Note:** A valid HuggingFace token (`HF_TOKEN`) is required for diarization tests.
> Export the variable before running the test:
>
> ```bash
> export HF_TOKEN=your_hf_token
> ```

See the test file for detailed comments on the pipeline steps and configuration overrides.

## Troubleshooting

### No Segments Produced

- Ensure `hf_token` is set and has access to the PyAnnote model
- Verify input audio files exist at the paths in the manifest
- Check that `audio_item_id` is unique per entry

### GPU Out of Memory

- Reduce `stages.4.batch_size` (first-pass ASR alignment)
- Reduce `stages.2.segmentation_batch_size` (diarization)
- Reduce `stages.10.batch_size` (second-pass ASR, when present)
- Process fewer files per manifest
- See [GPU Memory Requirements](#gpu-memory-requirements) for per-model VRAM usage

### Slow Processing

- Ensure GPU-accelerated stages have `resources` with `gpus=1` (the default)
- Increase `resources.cpus` for CPU-bound stages
- Split large manifests and process in parallel
- See [Timing Estimates](#timing-estimates) for expected throughput

## Related Documentation

- [Audio Getting Started Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/audio.html)
- [ALM Data Pipeline Tutorial](../alm/)
- [FLEURS Dataset Tutorial](../fleurs/)
- [NeMo Curator Installation](https://docs.nvidia.com/nemo/curator/latest/get-started/installation.html)
