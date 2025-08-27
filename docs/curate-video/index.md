---
description: "Comprehensive guide to Ray-based video curation with NeMo Curator including splitting and deduplication pipelines for large-scale processing"
categories: ["video-curation"]
tags: ["video-processing", "gpu-accelerated", "pipeline", "distributed", "ray", "splitting", "deduplication", "autoscaling"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "video-only"
---

(video-overview)=

# About Video Curation

Video curation is the process of taking long-form video content and dividing it into short, semantically consistent clips that you can filter for your use case.
Depending on the use case, this can involve processing 100+ PB of videos.
To efficiently process this quantity of videos, NeMo Curator provides highly optimized curation pipelines.

## Use Cases

* Generating clips for video world model training
* Generating clips for generative video model fine-tuning
* Creating a rich video database for video retrieval applications

## Architecture

```{image} ../about/concepts/video/_images/video-pipeline-diagram.png
:alt: High-level outline of NeMo Curator's video curation architecture
```

This diagram outlines NeMo Curator's video curation architecture.
Pipelines read videos from local or S3-compatible storage and write curated outputs to local storage.

Execution runs on Ray using our executor with streaming mode, per-stage resource settings (CPU, optional GPU, NVDEC/NVENC), and automatic scaling to balance workers across stages.

---

## Introduction

Master the fundamentals of NeMo Curator and set up your video processing environment.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Concepts
:link: about-concepts-video
:link-type: ref
Learn about the architecture, stages, pipelines, and data flow for video curation
+++
{bdg-secondary}`stages`
{bdg-secondary}`pipelines`
{bdg-secondary}`ray`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: gs-video
:link-type: ref
Install NeMo Curator, configure storage, prepare data, and run your first video pipeline.
:::

::::

---

## Curation Tasks

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Load Data
:link: video-load-data
:link-type: ref
Read videos from local or S3-compatible storage, or supply an explicit file list.
+++
{bdg-secondary}`local`
{bdg-secondary}`s3`
{bdg-secondary}`file-list`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Process Data
:link: video-process-data
:link-type: ref
Split into clips, encode, generate embeddings or captions, and remove duplicates.
+++
{bdg-secondary}`splitting`
{bdg-secondary}`embeddings`
{bdg-secondary}`captioning`
{bdg-secondary}`dedup`
:::

:::{grid-item-card} {octicon}`device-camera;1.5em;sd-mr-1` Save & Export
:link: video-save-export
:link-type: ref
Understand output directories, parquet embeddings, and packaging for training.
+++
{bdg-secondary}`parquet`
{bdg-secondary}`webdataset`
{bdg-secondary}`metadata`
:::

::::

---

## Tutorials

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Beginner Tutorial
:link: video-tutorials-beginner
:link-type: ref
Learn how to customize NeMo Curator's pipelines for your specific needs.
+++
{bdg-secondary}`video-splitting`
{bdg-secondary}`embeddings`
{bdg-secondary}`captioning`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Pipeline Customization Tutorials
:link: video-tutorials-pipeline-cust-series
:link-type: ref
Learn how to customize NeMo Curator's pipelines for your specific needs.
+++
{bdg-secondary}`custom-pipelines`
{bdg-secondary}`stages`
{bdg-secondary}`ray`
:::

::::
