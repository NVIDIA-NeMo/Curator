---
name: audio
description: |
  Process audio datasets with ASR transcription and quality filtering.
  Use when the user wants to transcribe audio, filter by WER (Word Error Rate),
  or prepare speech datasets. Supports NeMo ASR models.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: audio
  gpu-required: true
  nemo-curator-version: ">=0.5.0"
compatibility: Requires Python 3.10+, Linux, GPU with 16GB memory, NeMo Framework
disable-model-invocation: true
---

# Audio Processing

Process audio datasets with ASR transcription and Word Error Rate filtering.

## Platform Requirements

> **IMPORTANT**: NeMo Curator only runs on **Linux**. macOS and Windows are not supported.

## When to Use

- Transcribing audio datasets with ASR
- Filtering audio by transcription quality (WER)
- Preparing speech datasets for training
- Quality assessment of audio transcriptions

## Quick Start

### 1. List Available ASR Models

```bash
python scripts/list_asr_models.py

# Filter by language
python scripts/list_asr_models.py --language en
```

### 2. List Audio Stages

```bash
python scripts/list_audio_stages.py --verbose
```

### 3. Generate Pipeline Config

```bash
python scripts/generate_audio_config.py \
  --input-path /data/audio \
  --output-path /data/filtered \
  --model-name nvidia/stt_en_fastconformer_hybrid_large_pc \
  --wer-threshold 50 \
  --output-file audio_pipeline.yaml
```

### 4. Execute Pipeline

```bash
python -m nemo_curator.config.run \
  --config-path=. \
  --config-name=audio_pipeline
```

## Audio Processing Workflow

```
+------------------------------------------------------------------+
|                    Audio Curation Pipeline                        |
+------------------------------------------------------------------+
|                                                                   |
|  1. CREATE MANIFEST                                              |
|     - CreateInitialManifestFleursStage (for FLEURS dataset)      |
|     - OR provide custom manifest                                 |
|                                                                   |
|  2. ASR INFERENCE                                                |
|     - InferenceAsrNemoStage (NeMo ASR, GPU 16GB)                 |
|                                                                   |
|  3. QUALITY METRICS                                              |
|     - GetPairwiseWerStage (calculate WER)                        |
|     - GetAudioDurationStage (extract duration)                   |
|                                                                   |
|  4. FILTERING                                                    |
|     - PreserveByValueStage (filter by WER threshold)             |
|                                                                   |
|  5. CONVERT & WRITE                                              |
|     - AudioToDocumentStage (convert to text format)              |
|     - JsonlWriter (write output)                                 |
|                                                                   |
+------------------------------------------------------------------+
```

## Available Stages

| Stage | GPU | Description |
|-------|-----|-------------|
| CreateInitialManifestFleursStage | No | Prepare FLEURS dataset |
| InferenceAsrNemoStage | 16GB | Run ASR transcription |
| GetPairwiseWerStage | No | Calculate Word Error Rate |
| GetAudioDurationStage | No | Extract audio duration |
| PreserveByValueStage | No | Filter by field value |
| AudioToDocumentStage | No | Convert to document format |

## WER Threshold Guide

Word Error Rate (WER) measures transcription accuracy:

| WER | Quality | Use Case |
|-----|---------|----------|
| 0-10% | Excellent | High-quality training data |
| 10-25% | Good | General training |
| 25-50% | Moderate | Large-scale datasets |
| 50-75% | Poor | May need review |
| 75%+ | Bad | Usually filtered out |

### Recommended Thresholds

| Dataset Quality Goal | WER Threshold |
|---------------------|---------------|
| Premium | 15 |
| Standard | 50 |
| Permissive | 75 |

## Pipeline Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-name` | NeMo ASR model | Required |
| `--wer-threshold` | Max WER to keep | 75 |
| `--lang` | Language code | en |
| `--gpus` | GPUs for ASR | 1.0 |

## Common Workflows

### Basic ASR + Filtering

```bash
python scripts/generate_audio_config.py \
  --input-path /data/audio \
  --output-path /data/transcribed \
  --model-name nvidia/stt_en_fastconformer_hybrid_large_pc \
  --wer-threshold 50 \
  --output-file asr_pipeline.yaml
```

### High-Quality Dataset

```bash
python scripts/generate_audio_config.py \
  --input-path /data/audio \
  --output-path /data/high_quality \
  --model-name nvidia/stt_en_fastconformer_hybrid_large_pc \
  --wer-threshold 15 \
  --output-file quality_pipeline.yaml
```

## GPU Requirements

| Configuration | GPU Memory |
|---------------|------------|
| ASR Inference | 16GB |
| Full Pipeline | 16GB |

## References

- [ASR_MODELS.md](references/ASR_MODELS.md) - Available NeMo ASR models

## Related Skills

- `/curate` - Full curation workflow
- `/stages` - All available stages
- `/setup` - Environment setup
