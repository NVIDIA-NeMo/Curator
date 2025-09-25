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

- `AudioBatch` file existence checks using `validate()` and `validate_item()`
- Manifest format validation and metadata consistency
- Recommended JSONL manifest format

**Optional ASR Inference**:

- `InferenceAsrNemoStage` for automatic speech recognition
- Configurable batch processing with `batch_size` and `resources` parameters
- Support for multiple NeMo ASR models

**Quality Assessment**:

- Audio duration analysis with `GetAudioDurationStage`
- Word Error Rate (WER) and Character Error Rate (CER) calculation
- Speech rate metrics including words per second and characters per second

**Filtering and Quality Control**:

- Threshold-based filtering using `PreserveByValueStage`
- Configurable quality thresholds for WER, duration, and speech rate

**Export and Format Conversion**:

- Audio-to-text conversion with `AudioToDocumentStage`
- Integration with text processing workflows

## Common Workflows

**ASR-First Workflow** (Most Common):
1. Load audio files into `AudioBatch` format
2. Apply ASR inference to generate transcriptions
3. Calculate quality metrics (WER, duration, speech rate)
4. Apply threshold-based filtering
5. Convert to `DocumentBatch` for text processing integration
6. Export filtered, high-quality audio-text pairs

**Quality-First Workflow** (No ASR Required):
1. Load audio files with existing transcriptions
2. Extract audio characteristics (duration, format, sample rate)
3. Apply basic quality filters
4. Export validated audio dataset
