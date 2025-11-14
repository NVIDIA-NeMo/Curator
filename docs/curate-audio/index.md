---
description: "Comprehensive audio curation capabilities for speech data processing including ASR inference, quality assessment, and text integration workflows"
categories: ["workflows"]
tags: ["audio-curation", "asr-inference", "speech-processing", "quality-metrics", "manifests", "text-integration"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "workflow"
modality: "audio-only"
---

(audio-overview)=

# About Audio Curation

NeMo Curator provides comprehensive audio curation capabilities to prepare high-quality speech data for automatic speech recognition (ASR) and multi-modal model training. The toolkit includes processors for loading audio datasets, performing ASR inference, assessing transcription quality, and integrating with text curation workflows.

## Use Cases

- Process and curate large-scale speech datasets for ASR model training
- Perform quality assessment and filtering based on transcription accuracy metrics
- Generate transcriptions using state-of-the-art NVIDIA NeMo ASR models
- Integrate audio processing with text curation pipelines for multi-modal workflows
- Scale audio processing across GPU clusters efficiently

---

## Introduction

Master the fundamentals of NeMo Curator and set up your audio processing environment.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Concepts
:link: about-concepts-audio
:link-type: ref
Learn about AudioBatch, ASR pipelines, and other core data structures for efficient audio curation
+++
{bdg-secondary}`data-structures`
{bdg-secondary}`asr-pipeline`
{bdg-secondary}`quality-metrics`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: gs-audio
:link-type: ref
Learn prerequisites, setup instructions, and initial configuration for audio curation
+++
{bdg-secondary}`setup`
{bdg-secondary}`configuration`
{bdg-secondary}`quickstart`
:::

::::

## Curation Tasks

### Load Data

Import your audio data from various sources into NeMo Curator's processing pipeline.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`file-media;1.5em;sd-mr-1` Local Files
:link: audio-load-data-local
:link-type: ref
Load audio files from local directories and file systems
+++
{bdg-secondary}`local-storage`
{bdg-secondary}`file-discovery`
{bdg-secondary}`batch-processing`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Custom Manifests
:link: audio-load-data-custom-manifests
:link-type: ref
Create and load custom audio dataset manifests with metadata
+++
{bdg-secondary}`manifests`
{bdg-secondary}`metadata`
{bdg-secondary}`custom-formats`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` FLEURS Dataset
:link: audio-load-data-fleurs
:link-type: ref
Load and process the multilingual FLEURS speech dataset
+++
{bdg-secondary}`fleurs`
{bdg-secondary}`multilingual`
{bdg-secondary}`benchmarks`
:::

::::

### Process Data

Transform and enhance your audio data through ASR inference, quality assessment, and analysis.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` ASR Inference
:link: audio-process-data-asr-inference
:link-type: ref
Generate transcriptions using NVIDIA NeMo ASR models
+++
{bdg-secondary}`nemo-models`
{bdg-secondary}`transcription`
{bdg-secondary}`gpu-accelerated`
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Quality Assessment
:link: audio-process-data-quality-assessment
:link-type: ref
Assess transcription quality using WER and CER
+++
{bdg-secondary}`wer-filtering`
{bdg-secondary}`duration-filtering`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Audio Analysis
:link: audio-process-data-audio-analysis
:link-type: ref
Analyze audio characteristics including duration and format validation
+++
{bdg-secondary}`duration-calculation`
{bdg-secondary}`format-validation`
{bdg-secondary}`metadata-extraction`
:::

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Text Integration
:link: audio-process-data-text-integration
:link-type: ref
Integrate audio processing results with text curation workflows
+++
{bdg-secondary}`multimodal`
{bdg-secondary}`text-filtering`
{bdg-secondary}`pipeline-integration`
:::

::::

### Save & Export

Save processed audio data and transcriptions in formats suitable for downstream training and analysis.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`device-camera;1.5em;sd-mr-1` Save & Export
:link: audio-save-export
:link-type: ref
Export curated audio datasets with transcriptions and quality metrics
+++
{bdg-secondary}`manifests`
{bdg-secondary}`parquet`
{bdg-secondary}`metadata`
:::

::::

---

## Tutorials

Build practical experience with step-by-step guides for common audio curation workflows.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Beginner Tutorial
:link: audio-tutorials-beginner
:link-type: ref
Learn the basics of audio loading, ASR inference, and quality filtering
+++
{bdg-secondary}`asr-inference`
{bdg-secondary}`quality-filtering`
{bdg-secondary}`basic-workflow`
:::

::::
