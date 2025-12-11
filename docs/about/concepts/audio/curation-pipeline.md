---
description: "Modality-level overview of ingest, validation, optional ASR, metrics, filtering, and export"
categories: ["concepts-architecture"]
tags: ["audio-pipeline", "overview", "ingest", "metrics", "filtering", "export"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "audio-only"
---

(about-concepts-audio-curation-pipeline)=

# Audio Curation Pipeline (Overview)

This guide provides an overview of the end-to-end audio curation workflow in NVIDIA NeMo Curator. It covers data ingestion and validation, optional ASR inference, quality assessment, filtering, and export or conversion. For detailed ASR pipeline information, refer to {ref}`about-concepts-audio-asr-pipeline`.

## High-Level Flow

```{mermaid}
graph TD
    A[Audio Files] --> B[Ingest & Validation]
    B --> C[Optional ASR Inference]
    C --> D[Quality Metrics]
    B --> D
    D --> E[Filtering]
    E --> F[Export & Conversion]
```

## Core Components

**Data Ingestion and Validation**:

- `AudioBatch` file existence validation using `validate()` and `validate_item()`
- Recommended JSONL manifest format with audio file paths

**Optional ASR Inference**:

- `InferenceAsrNemoStage` for automatic speech recognition
- Configurable batch processing with `batch_size` and `resources` parameters
- Support for various NeMo ASR models

**Quality Assessment**:

- Audio duration analysis with `GetAudioDurationStage`
- Word Error Rate (WER) calculation with `GetPairwiseWerStage`
- Character Error Rate (CER) calculation using `get_cer()` function
- Speech rate metrics: `get_wordrate()` (words per second) and `get_charrate()` (characters per second)

**Filtering and Quality Control**:

- Threshold-based filtering using `PreserveByValueStage`
- Configurable quality thresholds for WER, duration, and speech rate

**Export and Format Conversion**:

- Audio-to-text conversion with `AudioToDocumentStage`
- Integration with text processing workflows

## Common Workflows

**ASR-First Workflow** (Most Common):

1. Load audio files into `AudioBatch` format
2. Apply ASR inference with `InferenceAsrNemoStage` to generate transcriptions
3. Calculate quality metrics: duration (`GetAudioDurationStage`), WER (`GetPairwiseWerStage`), speech rate
4. Apply threshold-based filtering with `PreserveByValueStage`
5. Convert to `DocumentBatch` with `AudioToDocumentStage` for text processing integration
6. Export filtered, high-quality audio-text pairs

**Quality-First Workflow** (No ASR Required):

1. Load audio files with existing transcriptions
2. Extract audio characteristics (duration, format, sample rate)
3. Apply basic quality filters
4. Export validated audio dataset
