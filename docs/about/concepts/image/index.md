---
description: "Essential concepts for image data curation including loading, processing, and export with GPU acceleration"
categories: ["concepts-architecture"]
tags: ["concepts", "image-curation", "webdataset", "gpu-accelerated", "embedding", "classification"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "image-only"
---

(about-concepts-image)=
# Image Curation Concepts

This document covers the essential concepts for image data curation in NVIDIA NeMo Curator. These concepts assume basic familiarity with data science and machine learning principles.

## Core Concept Areas

Image curation in NVIDIA NeMo Curator focuses on these key areas:

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data Loading
:link: about-concepts-image-data-loading
:link-type: ref

Core concepts for loading and managing image datasets
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Data Processing
:link: about-concepts-image-data-processing
:link-type: ref

Concepts for embedding generation, classification, filtering, and deduplication
:::

:::{grid-item-card} {octicon}`device-camera;1.5em;sd-mr-1` Data Export
:link: about-concepts-image-data-export
:link-type: ref

Concepts for saving, exporting, and resharding curated image datasets
:::

::::

## Infrastructure Components

The image curation concepts build on NVIDIA NeMo Curator's core infrastructure components, which are shared across all modalities (text, image, video). These components include:

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Memory Management
:link: reference-infra-memory-management
:link-type: ref
Optimize memory usage when processing large datasets
+++
{bdg-secondary}`partitioning`
{bdg-secondary}`batching`
{bdg-secondary}`monitoring`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` GPU Acceleration
:link: reference-infra-gpu-processing
:link-type: ref
Leverage NVIDIA GPUs for faster data processing
+++
{bdg-secondary}`cuda`
{bdg-secondary}`dali`
{bdg-secondary}`performance`
:::

:::{grid-item-card} {octicon}`sync;1.5em;sd-mr-1` Resumable Processing
:link: reference-infra-resumable-processing
:link-type: ref
Continue interrupted operations across large datasets
+++
{bdg-secondary}`checkpoints`
{bdg-secondary}`recovery`
{bdg-secondary}`batching`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Data Loading <data-loading-concepts>
Data Processing <data-processing-concepts>
Data Export <data-export-concepts>
```
