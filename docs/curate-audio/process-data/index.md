---
description: "Process audio data using ASR inference, quality assessment, audio analysis, and text integration for high-quality speech datasets"
categories: ["workflows"]
tags: ["audio-processing", "asr-inference", "quality-assessment", "audio-analysis", "text-integration", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "audio-only"
---

# Process Data for Audio Curation

Process audio data you've loaded into `AudioBatch` objects using NeMo Curator's comprehensive audio processing capabilities.

NeMo Curator provides a specialized suite of tools for processing speech and audio data as part of the AI training pipeline. These tools help you transcribe, analyze, filter, and integrate audio datasets to ensure high-quality input for ASR model training and multimodal applications.

## How it Works

NeMo Curator's audio processing capabilities are organized into four main categories:

1. **ASR Inference**: Transcribe audio using NeMo Framework's pretrained ASR models
2. **Quality Assessment**: Calculate and filter based on transcription accuracy metrics
3. **Audio Analysis**: Extract audio characteristics like duration and validate formats
4. **Text Integration**: Convert processed audio data to text processing workflows

Each category provides GPU-accelerated implementations optimized for different speech curation needs. The result is a cleaned and filtered audio dataset with high-quality transcriptions ready for model training.

---

## ASR Inference

Transcribe audio files using NeMo Framework's state-of-the-art ASR models with GPU acceleration.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` NeMo ASR Models
:link: asr-inference/nemo-models
:link-type: doc
Use pretrained NeMo ASR models for accurate speech recognition
+++
{bdg-secondary}`pretrained`
{bdg-secondary}`multilingual`
{bdg-secondary}`gpu-accelerated`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Batch Processing
:link: asr-inference/index
:link-type: doc
Efficiently process large audio datasets with configurable batch sizes
+++
{bdg-secondary}`batch-inference`
{bdg-secondary}`memory-optimization`
{bdg-secondary}`scalable`
:::

::::

## Quality Assessment

Evaluate and filter audio quality using transcription accuracy and audio characteristics.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` WER Filtering
:link: quality-assessment/wer-filtering
:link-type: doc
Filter audio samples based on Word Error Rate thresholds
+++
{bdg-secondary}`accuracy`
{bdg-secondary}`quality-metrics`
{bdg-secondary}`filtering`
:::

:::{grid-item-card} {octicon}`clock;1.5em;sd-mr-1` Duration Filtering
:link: quality-assessment/duration-filtering
:link-type: doc
Filter audio samples by duration ranges and speech rate metrics
+++
{bdg-secondary}`duration`
{bdg-secondary}`speech-rate`
{bdg-secondary}`range-filtering`
:::


::::

## Audio Analysis

Extract and analyze audio file characteristics for quality control and metadata generation.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`clock;1.5em;sd-mr-1` Duration Calculation
:link: audio-analysis/duration-calculation
:link-type: doc
Calculate precise audio duration using soundfile library
+++
{bdg-secondary}`soundfile`
{bdg-secondary}`precision`
{bdg-secondary}`metadata`
:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Format Validation
:link: audio-analysis/format-validation
:link-type: doc
Validate audio file formats and detect corrupted files
+++
{bdg-secondary}`validation`
{bdg-secondary}`error-handling`
{bdg-secondary}`format-support`
:::

::::

## Text Integration

Convert processed audio data to text processing workflows for multimodal applications.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Audio-to-Text Conversion
:link: text-integration/index
:link-type: doc
Convert AudioBatch objects to DocumentBatch for text processing
+++
{bdg-secondary}`format-conversion`
{bdg-secondary}`pipeline-integration`
{bdg-secondary}`multimodal`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

ASR Inference <asr-inference/index.md>
Quality Assessment <quality-assessment/index.md>
Audio Analysis <audio-analysis/index.md>
Text Integration <text-integration/index.md>
```
