---
description: "Essential concepts for text data curation including loading, processing, and synthetic data generation"
categories: ["concepts-architecture"]
tags: ["concepts", "text-curation", "data-processing", "synthetic-data", "distributed"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "text-only"
---

(about-concepts-text)=
# Text Curation Concepts

This document covers the essential concepts for text data curation in NVIDIA NeMo Curator. These concepts assume basic familiarity with data science and machine learning principles.

## Core Concept Areas

Text curation in NVIDIA NeMo Curator focuses on these key areas:

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Text Curation Pipeline
:link: about-concepts-text-data-curation-pipeline
:link-type: ref

Comprehensive overview of the end-to-end text curation architecture and workflow
+++
{bdg-secondary}`overview` {bdg-secondary}`architecture`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data Loading
:link: about-concepts-text-data-loading
:link-type: ref

Core concepts for loading and managing text datasets from local files
+++
{bdg-secondary}`local-files` {bdg-secondary}`formats`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Data Acquisition
:link: about-concepts-text-data-acquisition
:link-type: ref

Components for downloading and extracting data from remote sources
+++
{bdg-secondary}`remote-sources` {bdg-secondary}`download`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Data Processing
:link: about-concepts-text-data-processing
:link-type: ref

Concepts for filtering, deduplication, and classification
+++
{bdg-secondary}`filtering` {bdg-secondary}`quality`
:::


::::

## Infrastructure Components

The text curation concepts build on NVIDIA NeMo Curator's core infrastructure components, which are shared across all modalities. These components include:

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
{bdg-secondary}`rmm`
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

Curation Pipeline <data-curation-pipeline>
Data Loading <data-loading-concepts>
Data Acquisition <data-acquisition-concepts>
Data Processing <data-processing-concepts>
```
