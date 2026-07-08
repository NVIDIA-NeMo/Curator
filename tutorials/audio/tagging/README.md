# Audio Tagging Pipeline

This tutorial demonstrates how to process raw, unlabelled audio into labelled training data using NeMo Curator's audio tagging stages.

## Overview

The audio tagging pipeline is a generic processing framework that takes raw audio files and produces segmented, annotated manifests suitable for training multiple speech modalities — **TTS**, **ASR**, **ALM**, and others. The core pipeline (stages 0–9) is shared across all modalities: resampling, speaker diarization, ASR forced alignment, merge, quality metrics, and segment preparation. The `PrepareModuleSegmentsStage` is the key stage where segments are shaped differently based on the target modality (e.g. duration constraints, utterance completeness). Optionally, a second-pass ASR transcription and WER computation can be appended to further validate transcript quality.

### Pipeline Flow

```
 ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
 │     Manifest     │───▶│     Resample     │───▶│     Diarize      │───▶│    Split Long    │
 │      Reader      │    │   (16kHz WAV)    │    │    (PyAnnote)    │    │      Audio       │
 └──────────────────┘    └──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                                                  │
                                                                                  ▼
 ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
 │      Merge       │◀───│    Join Split    │◀───│    ASR Align     │
 │    Align+Diar    │    │     Metadata     │    │   (1st pass)     │
 └────────┬─────────┘    └──────────────────┘    └──────────────────┘
          │
          ▼
 ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
 │    Bandwidth     │───▶│      SQUIM       │───▶│     Prepare      │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
 │    Estimation    │    │     Metrics      │    │    Module Seg    │    (tts / asr / ...)
 └──────────────────┘    └──────────────────┘    └────────┬─────────┘                            │
                                                          │
                                                          ▼                                      │
                                                 ┌──────────────────┐
                                                 │    ASR Align     │                            │
                                                 │    (2nd pass)    │
                                                 └────────┬─────────┘                            │
                                                          │
                                                          ▼                                      │
                                                 ┌────────────────┐
                                                 │  Compute WER   │                              │
                                                 │                │
                                                 └───────┬────────┘                              │
                                                         │
                                                         ▼                                       │
                                                 ┌────────────────┐
                                                 │    Manifest    │◀─────────────────────────────┘
                                                 │     Writer     │
                                                 └────────────────┘
```

The dashed path shows that `ManifestWriter` can follow directly after `PrepareModuleSegments` (the default TTS tutorial config) or after the second-pass ASR + WER stages (the ASR tutorial config). The nightly TTS benchmark deliberately follows the full second-pass path so stage coverage cannot regress silently.

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
| 9 | **PrepareModuleSegmentsStage** | Merge/split segments into training-ready chunks for the target modality. Uses `min_duration` and `max_duration` (in seconds) to form segments suitable for ASR/TTS training. Controlled by the `module` parameter (`tts`, `asr`, etc.) and also considers pauses and punctuation for splitting. | No |

> **Punctuation matters**: `PrepareModuleSegmentsStage` relies heavily on punctuation marks (`.`, `!`, `?`) to identify natural utterance boundaries when forming segments. If the ASR model produces unpunctuated text, segments will be split purely by duration and pause heuristics, leading to mid-sentence breaks. Use an ASR model that outputs punctuated and capitalised text natively for best results.

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

## Installation

From the Curator repository root:

```bash
uv sync --extra audio_cuda12
source .venv/bin/activate
```

### Prerequisites

- **System packages**: `ffmpeg` must be installed for audio resampling and format conversion:
  ```bash
  # Ubuntu / Debian
  sudo apt-get install -y ffmpeg

  ```
- **GPU**: Required for diarization (PyAnnote), VAD (Pyannote), ASR alignment (NeMo)
- **HuggingFace Token**: Required for PyAnnote diarization model access. See [HuggingFace Access](#huggingface-access) for setup instructions.

## Quick Start

### TTS Pipeline

The TTS config runs the core stages with `module: tts` in `PrepareModuleSegmentsStage` (`full_utterance_ratio: 1.0`). The output segments are single-speaker utterances, each annotated with quality metrics such as `bandwidth`, `stoi_squim`, `sisdr_squim`, and `pesq_squim`. These metrics can be used downstream to filter for high-quality audio — for example, keeping only segments where `bandwidth >= 8000 && sisdr_squim >= 15 && stoi_squim >= 0.9`.

A small toy dataset is bundled in `tests/fixtures/audio/tagging/` so you can run end-to-end without providing your own audio:

```bash
read -rsp "Hugging Face token: " HF_TOKEN && export HF_TOKEN
echo
python tutorials/audio/tagging/main.py \
  --config-path . \
  --config-name tts_pipeline \
  input_manifest=tests/fixtures/audio/tagging/sample_input.jsonl \
  final_manifest=/tmp/tts_output.jsonl
```

### ASR Pipeline

The ASR config runs the same core stages with `module: asr` (`full_utterance_ratio: 0.8` to allow partial utterances), then adds second-pass ASR and WER computation. The per-segment `metrics.wer.wer` ratio can be used to filter for reliable transcripts, for example by keeping only segments where `metrics.wer.wer <= 0.10`.

```bash
python tutorials/audio/tagging/main.py \
  --config-path . \
  --config-name asr_pipeline \
  input_manifest=/data/input.jsonl \
  final_manifest=/data/asr_output.jsonl
```

#### Improving ASR Training Data Quality

For ASR training data, combine these optional blocks to maximise transcript quality:

1. **Filter by WER**: After the second-pass ASR and `ComputeWERStage`, filter segments with `metrics.wer.wer <= 0.10` to keep only samples where the two ASR passes agree closely. This is a strong signal that the transcript is correct.
2. **Apply ITN**: Insert `InverseTextNormalizationStage` to convert spoken-form text (e.g. "twenty three") to written form (e.g. "23") for training data that requires normalised text.

These blocks compose naturally — ITN and WER filtering each address a different axis of data quality and can both be enabled in a single pipeline run.

### Nightly TTS Benchmark

`benchmarking/scripts/audio_tagging_benchmark.py` is stricter than the minimal
TTS tutorial config. It uses `module: tts` but always executes the second-pass
`ASRAlignment2` and `ComputeWER` stages before `ManifestWriter`:

```text
FLEURS or ManifestReader -> ResampleAudio -> PyAnnoteDiarization
  -> SplitLongAudio -> ASRAlignment -> JoinSplitMetadata
  -> MergeAlignmentDiar -> BandwidthEstimation -> SquimMetrics
  -> PrepareModuleSegments -> ASRAlignment2 -> ComputeWER -> ManifestWriter
```

The nightly input is the complete English FLEURS development split: 394 clips
and approximately 1.05 hours of audio. The repeated entry processes 788 tasks
and approximately 2.09 source audio hours. Both entries use a pre-staged copy
and pass `--no-auto-download`, so dataset download time is outside the measured
run.

The benchmark reports success only after validating non-empty prepared output,
100% second-pass ASR and WER coverage across valid segments, nonzero work from
all 12 processing stages, and a persisted JSONL row for every output task. See
the [benchmarking guide](../../../benchmarking/README.md#audio-tagging-benchmark)
for staging, direct-run, metric, and pass/fail details.

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

The runner validates its result before printing `PIPELINE COMPLETE`. At least one output task and one structurally valid prepared segment are required; each counted segment must contain `speaker`, finite ordered `start`/`end`, non-empty `text` and `words`, and a `metrics` mapping.

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
      "text_2": "Hello, how are you today?",
      "metrics": {
        "bandwidth": [8000, 8400],
        "pesq_squim": [3.4, 3.5],
        "stoi_squim": [0.91, 0.92],
        "sisdr_squim": [19.8, 20.4],
        "wer": {"wer": 0.0, "tokens": 5, "ins_rate": 0.0, "del_rate": 0.0, "sub_rate": 0.0}
      }
    }
  ],
  "overlap_segments": [],
  "text": "Hello, how are you today? Let's get started with the tutorial.",
  "alignment": [
    {"word": "Hello", "start": 1.23, "end": 1.55},
    {"word": "how", "start": 1.60, "end": 1.72}
  ]
}
```

`text_2` and the WER metrics are present only when the second-pass stages are
configured. The nightly benchmark requires both on every prepared segment.

### Output Fields

| Field                     | Source                  | Description                                                          |
|---------------------------|-------------------------|----------------------------------------------------------------------|
| `resampled_audio_filepath`| Core                    | Path to the resampled 16 kHz mono WAV                                |
| `duration`                | Core                    | Total audio duration in seconds                                      |
| `segments`                | Core                    | List of labelled speaker segments with text, word timestamps         |
| `overlap_segments`        | Core                    | Speaker turns with detected overlap (excluded from `segments`)       |
| `text`                    | Core                    | Full transcript text for the audio entry                             |
| `alignment`               | Core                    | List of word-level alignment objects (`word`, `start`, `end`)        |
| `segments[].metrics.bandwidth`    | Core                    | Per-word estimated spectral bandwidth values                         |
| `segments[].metrics.pesq_squim`   | Core                    | Per-word PESQ quality scores (via TorchSQUIM)                         |
| `segments[].metrics.stoi_squim`   | Core                    | Per-word STOI quality scores (via TorchSQUIM)                         |
| `segments[].metrics.sisdr_squim`  | Core                    | Per-word SI-SDR quality scores (via TorchSQUIM)                       |
| `segments[].text_2`       | Optional (2nd-pass ASR) | Second-pass ASR transcript (e.g. CTC Conformer)                     |
| `segments[].metrics.wer.wer` | Optional (ComputeWER) | Word error rate between first and second ASR transcripts             |

## Configuration

All parameters are defined in the YAML config files. Override from the command line:

```bash
python tutorials/audio/tagging/main.py \
  --config-path . \
  --config-name tts_pipeline \
  input_manifest=tests/fixtures/audio/tagging/sample_input.jsonl \
  final_manifest=/tmp/output.jsonl \
  language_short=de \
  max_segment_length=30
```

### Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_manifest` | Path to input JSONL manifest | **Required** |
| `final_manifest` | Path for output JSONL manifest | **Required** |
| `HF_TOKEN` | Environment variable for PyAnnote access (see [HuggingFace Access](#huggingface-access) below) | **Required** |
| `sample_rate` | Target sample rate in Hz | `16000` |
| `max_segment_length` | Maximum segment duration in seconds | `40` |
| `workspace_dir` | Directory for intermediate files | `/tmp/tagging_workspace` |
| `resampled_audio_dir` | Directory for resampled audio | `${workspace_dir}/audio_resampled` |

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
├── text/
│   ├── test_itn.py
│   └── test_text.py
└── e2e/
    ├── test_tts_e2e.py
    ├── test_asr_e2e.py
    ├── conftest.py
    ├── utils.py
    └── configs/
        ├── tts_pipeline.yaml
        └── asr_pipeline.yaml
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
├── test_tts_e2e.py             # End-to-end TTS tagging pipeline test
├── test_asr_e2e.py             # End-to-end ASR tagging pipeline test
├── conftest.py                 # Test fixtures (manifests, input data)
├── utils.py                    # Output validation helpers
└── configs/
    ├── tts_pipeline.yaml               # TTS pipeline configuration
    └── asr_pipeline.yaml               # ASR pipeline configuration
```

> **Note:** A valid HuggingFace token (`HF_TOKEN`) is required for diarization tests.
> Export the variable before running the test:
>
> ```bash
> read -rsp "Hugging Face token: " HF_TOKEN && export HF_TOKEN
> echo
> ```

See the test file for detailed comments on the pipeline steps and configuration overrides.

## HuggingFace Access

The pipeline requires a HuggingFace token with accepted user agreements for the PyAnnote speaker diarization models. To set up access:

1. Create a HuggingFace account at https://huggingface.co/join
2. Generate an access token at https://huggingface.co/settings/tokens
3. Accept the user agreements for the following models:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) — main diarization pipeline
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) — speaker segmentation model
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) — community diarization model
   - [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection) — voice activity detection
4. Read the token into `HF_TOKEN` without putting it in shell history or the process command line:

```bash
read -rsp "Hugging Face token: " HF_TOKEN && export HF_TOKEN
echo
```

The tutorial YAML reads `HF_TOKEN` through OmegaConf. Avoid overriding
`hf_token` on the command line because process listings and shell history may
retain the credential.

> **Note:** Without accepted agreements, the pipeline will fail with a 401/403 error when attempting to download the PyAnnote models.

## Troubleshooting

### No Segments Produced

- Ensure `HF_TOKEN` is set and has access to the PyAnnote models (see [HuggingFace Access](#huggingface-access))
- Verify input audio files exist at the paths in the manifest
- Check that `audio_item_id` is unique per entry
- Use a non-trivial audio sample long enough to satisfy `PrepareModuleSegmentsStage.min_duration`; zero valid prepared segments is a failed run and will not print `PIPELINE COMPLETE`
- Inspect diarization and ASR alignment output before `PrepareModuleSegmentsStage` when the validation error reports empty or malformed segments

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
