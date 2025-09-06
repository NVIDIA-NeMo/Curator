---
description: "Comprehensive audio curation capabilities for preparing high-quality speech datasets with ASR inference, quality assessment, and filtering"
categories: ["workflows"]
tags: ["audio-curation", "asr", "speech-recognition", "quality-filtering", "transcription", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "workflow"
modality: "audio-only"
---

(audio-overview)=
# About Audio Curation

NeMo Curator provides comprehensive audio curation capabilities to prepare high-quality speech datasets for automatic speech recognition (ASR) model training and multimodal applications. The toolkit includes processors for loading audio files, performing ASR inference, calculating quality metrics, and filtering based on transcription accuracy.

## Use Cases

- Prepare speech datasets for ASR model training and fine-tuning
- Quality assessment and filtering of existing speech corpora
- Create high-quality audio-text pairs for multimodal model training
- Process multilingual speech datasets with language-specific ASR models
- Filter and curate podcast, lecture, and conversational audio data

## Architecture

Audio curation in NeMo Curator follows a **Load** → **Process** → **Assess** workflow: load audio files and manifests, perform ASR inference using NeMo models, calculate quality metrics like Word Error Rate (WER), and filter based on transcription accuracy and audio characteristics.

---

## Introduction

Master the fundamentals of NeMo Curator's audio processing and set up your speech curation environment.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Concepts
:link: about-concepts-audio
:link-type: ref
Learn about AudioBatch and core data structures for efficient audio curation
+++
{bdg-secondary}`data-structures`
{bdg-secondary}`asr-pipeline`
{bdg-secondary}`architecture`
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

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` FLEURS Dataset
:link: audio-load-data-fleurs
:link-type: ref
Load and process the multilingual FLEURS speech dataset
+++
{bdg-secondary}`multilingual`
{bdg-secondary}`speech-corpus`
{bdg-secondary}`automated`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Custom Manifests
:link: audio-load-data-custom-manifests
:link-type: ref
Create and load custom audio manifests with file paths and transcriptions
+++
{bdg-secondary}`manifests`
{bdg-secondary}`custom-data`
{bdg-secondary}`jsonl`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Local Files
:link: audio-load-data-local
:link-type: ref
Load audio files from local directories and file systems
+++
{bdg-secondary}`local-files`
{bdg-secondary}`batch-processing`
{bdg-secondary}`file-validation`
:::

::::

### Process Data

Transform and enhance your audio data through ASR inference, quality assessment, and filtering.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` ASR Inference
:link: process-data/asr-inference/index
:link-type: doc
Perform automatic speech recognition using NeMo Framework models
+++
{bdg-secondary}`nemo-models`
{bdg-secondary}`gpu-accelerated`
{bdg-secondary}`batch-processing`
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Quality Assessment
:link: process-data/quality-assessment/index
:link-type: doc
Assess and filter audio quality using WER, duration, and custom metrics
+++
{bdg-secondary}`wer-filtering`
{bdg-secondary}`duration-filtering`
{bdg-secondary}`quality-metrics`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Audio Analysis
:link: process-data/audio-analysis/index
:link-type: doc
Extract audio characteristics like duration, format validation, and metadata
+++
{bdg-secondary}`duration-extraction`
{bdg-secondary}`format-validation`
{bdg-secondary}`metadata`
:::

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Text Integration
:link: process-data/text-integration/index
:link-type: doc
Convert processed audio data to text processing pipelines
+++
{bdg-secondary}`transcription`
{bdg-secondary}`text-processing`
{bdg-secondary}`multimodal`
:::

::::

### Save Data

Export processed audio data and transcriptions in formats for training and analysis.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Save & Export
:link: audio-save-export
:link-type: ref
Understand output formats, manifest structures, and export options for audio datasets
+++
{bdg-secondary}`jsonl`
{bdg-secondary}`manifests`
{bdg-secondary}`metadata`
:::

::::

---

## Tutorials

Practice with guided, hands-on examples to build, customize, and run audio curation pipelines.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Beginner Tutorial
:link: audio-tutorials-beginner
:link-type: ref
Create and run your first audio curation pipeline using the FLEURS dataset
+++
{bdg-secondary}`fleurs-dataset`
{bdg-secondary}`asr-inference`
{bdg-secondary}`quality-filtering`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Pipeline Customization Tutorials
:link: audio-tutorials-pipeline-cust-series
:link-type: ref
Customize ASR models, quality filters, and integration workflows
+++
{bdg-secondary}`custom-models`
{bdg-secondary}`custom-filters`
{bdg-secondary}`integration`
:::

::::
