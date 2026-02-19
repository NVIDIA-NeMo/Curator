# Audio Curation Packages

Reference for audio-specific NeMo Curator dependencies.

## Package Extras

| Extra | Use Case |
|-------|----------|
| `audio_cpu` | Audio processing without GPU |
| `audio_cuda12` | Full audio curation with GPU ASR |

**Note**: Audio extras have a transformers version conflict. Use override:

```bash
echo "transformers==4.55.2" > override.txt
uv pip install nemo-curator[audio_cuda12] --override override.txt
```

## Dependencies

### audio_cpu / audio_cuda12

| Package | Purpose |
|---------|---------|
| `nemo_toolkit[asr]` | NVIDIA NeMo ASR models |
| `torchaudio` | Audio processing |

## Available Stages

| Stage | Purpose | GPU Memory |
|-------|---------|------------|
| `InferenceAsrNemoStage` | Speech-to-text | 8-16 GB |
| `GetPairwiseWerStage` | Word Error Rate | CPU |

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | `.wav` | Uncompressed |
| MP3 | `.mp3` | Compressed |
| FLAC | `.flac` | Lossless compressed |

## GPU Requirements

| Stage | GPU Memory |
|-------|------------|
| ASR inference | 8-16 GB (model dependent) |
| WER calculation | CPU only |

**Minimum**: 8 GB VRAM
**Recommended**: 16 GB for larger ASR models

## Common Workflows

### Speech Transcription

```yaml
stages:
  - _target_: nemo_curator.stages.audio.inference.asr_nemo.InferenceAsrNemoStage
    model_name: "nvidia/parakeet-tdt-0.6b"
```

### Quality Filtering by WER

```yaml
stages:
  - _target_: nemo_curator.stages.audio.inference.asr_nemo.InferenceAsrNemoStage
  - _target_: nemo_curator.stages.audio.metrics.get_wer.GetPairwiseWerStage
    text_key: "text"
    pred_text_key: "pred_text"
```

## Common Issues

### Transformers Version Conflict

NeMo Toolkit requires specific transformers version:

```bash
echo "transformers==4.55.2" > override.txt
uv pip install nemo-curator[audio_cuda12] --override override.txt
```

### NeMo Toolkit Import Error

```bash
# If nemo.collections.asr fails
uv pip install nemo_toolkit[asr] --override override.txt
```

### Model Download Issues

NeMo models download from NGC. May need authentication for some models:

```bash
# Set NGC API key if needed
export NGC_API_KEY=your_key
```
