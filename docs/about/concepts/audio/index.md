---
description: "Essential concepts for audio data curation including ASR inference, quality assessment, and speech processing workflows"
categories: ["concepts-architecture"]
tags: ["concepts", "audio-curation", "asr", "speech-processing", "quality-metrics"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "audio-only"
---

(about-concepts-audio)=

# Audio Curation Concepts

This guide covers the essential concepts for audio data curation in NVIDIA NeMo Curator. These concepts assume basic familiarity with speech processing and machine learning principles.

## Core Concept Areas

Audio curation in NVIDIA NeMo Curator focuses on these key areas:

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Audio Curation Pipeline
:link: about-concepts-audio-curation-pipeline
:link-type: ref

Modality-level overview of ingest, validation, optional ASR, metrics, filtering, and export
+++
{bdg-secondary}`overview` {bdg-secondary}`map`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` ASR Pipeline
:link: about-concepts-audio-asr-pipeline
:link-type: ref

Comprehensive overview of the automatic speech recognition pipeline and workflow
+++
{bdg-secondary}`overview` {bdg-secondary}`architecture`
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Quality Metrics
:link: about-concepts-audio-quality-metrics
:link-type: ref

Core concepts for evaluating speech transcription quality and audio characteristics
+++
{bdg-secondary}`wer` {bdg-secondary}`cer` {bdg-secondary}`metrics`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` AudioBatch Structure
:link: about-concepts-audio-audio-batch
:link-type: ref

Understanding the AudioBatch data structure and audio file management
+++
{bdg-secondary}`data-structures` {bdg-secondary}`validation`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Dataset Manifests and Ingest
:link: about-concepts-audio-manifests-ingest
:link-type: ref

Concepts for constructing manifests and ingesting audio datasets
+++
{bdg-secondary}`manifests` {bdg-secondary}`ingest`
:::

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Text Integration
:link: about-concepts-audio-text-integration
:link-type: ref

Concepts for integrating audio processing with text curation workflows
+++
{bdg-secondary}`multimodal` {bdg-secondary}`integration`
:::

::::

## Infrastructure Components

The audio curation concepts build on NVIDIA NeMo Curator's core infrastructure components, which are shared across all modalities. These components include:

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Memory Management
:link: reference-infra-memory-management
:link-type: ref
Optimize memory usage when processing large audio datasets
+++
{bdg-secondary}`partitioning`
{bdg-secondary}`batching`
{bdg-secondary}`monitoring`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` GPU Acceleration
:link: reference-infra-gpu-processing
:link-type: ref
Leverage NVIDIA GPUs for faster ASR inference and audio processing
+++
{bdg-secondary}`cuda`
{bdg-secondary}`nemo-toolkit`
{bdg-secondary}`performance`
:::

:::{grid-item-card} {octicon}`sync;1.5em;sd-mr-1` Resumable Processing
:link: reference-infra-resumable-processing
:link-type: ref
Continue interrupted operations across large audio datasets
+++
{bdg-secondary}`checkpoints`
{bdg-secondary}`recovery`
{bdg-secondary}`batching`
:::

:::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Audio Curation Pipeline (Overview) <curation-pipeline.md>
ASR Pipeline <asr-pipeline.md>
Quality Metrics <quality-metrics.md>
AudioBatch Structure <audio-batch.md>
Dataset Manifests and Ingest <manifests-ingest.md>
Text Integration <text-integration.md>
```
