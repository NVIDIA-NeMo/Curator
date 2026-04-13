# VAD + ASR Pipeline

This tutorial runs the **voice activity detection (VAD) + ASR** path in NeMo Curator: raw audio is resampled, segmented with **WhisperX VAD**, aligned with **NeMo FastConformer** twice (RNNT and CTC), then **normalized WER / CER metrics** are attached per segment before writing a JSONL manifest.

It does **not** run speaker diarization, split/join long audio, or merge alignment with diarization (see the [audio tagging tutorial](../tagging/) for that flow).

## Overview

### Pipeline flow

```
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│ Raw Audio  │─▶│ Resample   │─▶│ VAD        │─▶│ ASR align  │─▶│ ASR align  │
│ Manifest   │  │ (16 kHz)   │  │ (WhisperX) │  │ (RNNT)     │  │ (CTC)      │
└────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘
                                                                        │
┌────────────┐  ┌────────────┐                                          │
│ Output     │◀─│ Manifest   │◀─ WER / CER metrics (per segment) ◀──────┘
│ JSONL      │  │ Writer     │
└────────────┘  └────────────┘
```

### Pipeline stages

| # | Stage | Description | GPU |
|---|-------|-------------|-----|
| 0 | **ManifestReader** | Reads input JSONL manifest | No |
| 1 | **ResampleAudioStage** | Resample to 16 kHz mono WAV | No |
| 2 | **WhisperXVADStage** | VAD segments via WhisperX / pyannote | Yes (default) |
| 3 | **NeMoASRAlignerStage** | Forced alignment, FastConformer RNNT; writes `text` | Yes |
| 4 | **NeMoASRAlignerStage** | Second pass, FastConformer CTC; writes `text2` | Yes |
| 5 | **ComputeNormalizedWERMetricsStage** | Normalized WER/CER (and optional PnC WER) on `text` vs `text2` per segment | No |
| 6 | **ManifestWriterStage** | Writes output JSONL manifest | No |

Configuration lives in `vad_asr_pipeline.yaml`.

## Installation

From the Curator repository root:

```bash
uv sync --extra audio_cuda12
source .venv/bin/activate
```

### Prerequisites

- **System packages**: `ffmpeg` and `sox` for resampling and format conversion:

  ```bash
  # Ubuntu / Debian
  sudo apt-get install -y ffmpeg sox libsox-fmt-all

  # macOS
  brew install ffmpeg sox
  ```

- **GPU**: Recommended for **WhisperXVADStage** and both **NeMoASRAlignerStage** instances (NeMo alignment loads GPU checkpoints by default when CUDA is available).

- **Hugging Face token** (`hf_token`): Often required for **gated pyannote** assets used under the hood by WhisperX VAD, and useful for NeMo checkpoint access. Accept the model terms on the Hub for any models you use (for example [pyannote/segmentation](https://huggingface.co/pyannote/segmentation-3.0) and related VAD/segmentation checkpoints referenced by your WhisperX install).

## Quick start

From `tutorials/audio/asr` (or pass an absolute `--config-path`):

```bash
python tutorials/audio/asr/main.py \
  --config-path tutorials/audio/asr \
  --config-name vad_asr_pipeline \
  input_manifest=/data/input.jsonl \
  final_manifest=/data/vad_asr_output.jsonl \
  hf_token=<your_hf_token>
```

Optional overrides (examples):

```bash
python tutorials/audio/asr/main.py \
  --config-path tutorials/audio/asr \
  --config-name vad_asr_pipeline \
  input_manifest=/data/input.jsonl \
  final_manifest=/data/out.jsonl \
  hf_token=<your_hf_token> \
  language_short=en \
  max_segment_length=30 \
  stages.3.batch_size=16 \
  stages.4.batch_size=16
```

## Input format

Each JSONL line should include at least:

```json
{
  "audio_filepath": "/path/to/raw/audio.wav",
  "audio_item_id": "unique_id_001"
}
```

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `audio_filepath` | string | Path to the raw audio file |
| `audio_item_id` | string | Unique identifier for the audio entry |

After resampling, downstream stages read `resampled_audio_filepath` produced by **ResampleAudioStage**.

## Output format

Each output line extends the entry with resampled paths, **VAD-derived `segments`**, two ASR hypotheses (**`text`** from RNNT, **`text2`** from CTC), word-level alignment fields from the aligners, and a **`metrics`** object per segment from **ComputeNormalizedWERMetricsStage** (WER, CER, optional punctuation-normalized metrics, rates, and so on—see `nemo_curator/stages/audio/metrics/get_wer.py`).

Illustrative shape (fields vary with model and segment content):

```json
{
  "audio_filepath": "/path/to/audio.wav",
  "audio_item_id": "unique_id_001",
  "resampled_audio_filepath": "/tmp/tagging_workspace/audio_resampled/unique_id_001.wav",
  "duration": 87.13,
  "segments": [
    {
      "start": 1.23,
      "end": 6.78,
      "text": "hello world from rnnt",
      "text2": "hello world from ctc",
      "words": [ ... ],
      "metrics": {
        "wer": { "wer": 0.0, "tokens": [], "ins_rate": 0.0, "del_rate": 0.0, "sub_rate": 0.0 },
        "cer": { ... }
      }
    }
  ]
}
```

### Output fields (high level)

| Field | Description |
|-------|-------------|
| `resampled_audio_filepath` | Path to 16 kHz mono WAV from resampling |
| `duration` | Total duration in seconds (when present) |
| `segments` | VAD segments with ASR `text`, `text2`, `words`, and optional `metrics` |
| `text` | Full-utterance or aggregated transcript key used by the pipeline (aligner / metrics config) |
| `text2` | Second-pass hypothesis from the CTC aligner stage |

## Configuration

Top-level keys in `vad_asr_pipeline.yaml` include:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_manifest` | Input JSONL manifest | **Required** |
| `final_manifest` | Output JSONL manifest | **Required** |
| `hf_token` | Hugging Face token string | `""` |
| `workspace_dir` | Intermediate workspace | `/tmp/tagging_workspace` |
| `resampled_audio_dir` | Resampled WAV output dir | `${workspace_dir}/audio_resampled` |
| `sample_rate` | Target sample rate (Hz) | `16000` |
| `max_segment_length` | VAD merge cap passed to WhisperX as `max_length` | `40` |
| `language_short` | Language code for text normalization in WER stage | `en` |
| `batch_size` | Declared in YAML (individual ASR stages set their own `batch_size` in the file) | `4` |
| `resources.cpus` | CPU resources for CPU-bound stages | `2` |

### Stage-specific overrides

Use the stage index in `stages` (see table above). Examples:

```bash
# Shorter VAD merge cap (stage 2)
stages.2.max_length=30

# RNNT aligner batch size (stage 3)
stages.3.batch_size=8

# CTC aligner batch size (stage 4)
stages.4.batch_size=8

# Skip very short segments for RNNT (stage 3)
stages.3.min_len=2.0
```

**WhisperXVADStage** (`stages.2`): `random_max_length: true` in the default YAML randomizes the internal merge cap when enabled (see `WhisperXVADModel` docstring for the exact sampling rule).

## File structure

```
tutorials/audio/asr/
├── main.py                 # Hydra entrypoint (VAD + ASR)
├── vad_asr_pipeline.yaml   # Pipeline definition
└── README.md               # This file
```

## Testing

There is no separate ASR-tutorial E2E in this repo; reuse tests for the underlying stages:

```bash
# VAD
pytest tests/stages/audio/inference/vad/ -v

# ASR alignment
pytest tests/stages/audio/tagging/inference/test_nemo_asr_align.py -v

# WER metrics
pytest tests/stages/audio/metrics/ -v
```

The full **tagging** pipeline E2E (diarization + alignment + merge) lives under `tests/stages/audio/tagging/e2e/` and is documented in [../tagging/README.md](../tagging/README.md).

## Troubleshooting

### Empty or missing `segments`

- Confirm **WhisperX VAD** weights load (CUDA OOM, missing deps, or HF gating).
- Ensure `hf_token` is valid if Hub access is required.
- Very short files are skipped by **WhisperXVADStage** when duration is below `min_length` (default 0.5 s).

### GPU out of memory

- Lower `stages.3.batch_size` and `stages.4.batch_size`.
- Reduce `max_segment_length` so VAD emits fewer long segments.

### Slow runs

- Keep GPU `resources` on VAD and ASR stages (defaults request GPU for VAD).
- Increase `resources.cpus` for resampling if CPU-bound.

## Related documentation

- [Audio tagging tutorial](../tagging/README.md) (diarization, split/join, merge)
- [Audio Getting Started Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/audio.html)
- [NeMo Curator installation](https://docs.nvidia.com/nemo/curator/latest/get-started/installation.html)
