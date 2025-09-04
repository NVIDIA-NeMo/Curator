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

Learn what video curation is and how you use NeMo Curator to turn long videos into high‑quality, searchable clips.
Depending on the use case, this can involve processing 100+ PB of videos.
To efficiently process this quantity of videos, NeMo Curator provides highly optimized curation pipelines.

## Use Cases

Identify when to use NeMo Curator by matching your goals to common video curation scenarios.

* Generating clips for video world model training
* Generating clips for generative video model fine-tuning
* Creating a rich video database for video retrieval applications

## Architecture

Understand how components work together so you can plan, scale, and troubleshoot video pipelines. The following diagram outlines NeMo Curator's video curation architecture:

```{image} ../about/concepts/video/_images/video-pipeline-diagram.png
:alt: High-level outline of NeMo Curator's video curation architecture
```

---

## Introduction

Get oriented and prepare your environment so you can start curating videos with confidence.

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

Follow task-based guides to load, process, and write curated video data end to end.

### Load Data

Bring videos into your pipeline from local paths or remote sources you control.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Local & Cloud
:link: video-load-data-local-cloud
:link-type: ref
Load videos from local paths or S3-compatible and HTTP(S) URLs.
+++
{bdg-secondary}`local`
{bdg-secondary}`s3`
{bdg-secondary}`file-list`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Remote (JSON)
:link: video-load-data-json-list
:link-type: ref
Provide an explicit JSON file list for remote datasets under a root prefix.
+++
{bdg-secondary}`file-list`
{bdg-secondary}`s3`
:::

::::

### Process Data

Transform raw videos into curated clips, frames, embeddings, and metadata you can use.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Clip Videos
:link: video-process-clipping
:link-type: ref
Split long videos into shorter clips using fixed stride or scene-change detection.
+++
{bdg-primary}`clips`
{bdg-secondary}`fixed-stride`
{bdg-secondary}`transnetv2`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Encode Clips
:link: video-process-transcoding
:link-type: ref
Encode clips to H.264 using CPU or GPU encoders and tune performance.
+++
{bdg-primary}`clips`
{bdg-secondary}`h264_nvenc`
{bdg-secondary}`libopenh264`
{bdg-secondary}`libx264`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Filter Clips and Frames
:link: video-process-filtering
:link-type: ref
Apply motion-based filtering and aesthetic filtering to improve dataset quality.
+++
{bdg-primary}`clips`
{bdg-primary}`frames`
{bdg-secondary}`motion`
{bdg-secondary}`aesthetic`
:::

:::{grid-item-card} {octicon}`device-camera;1.5em;sd-mr-1` Extract Frames
:link: video-process-frame-extraction
:link-type: ref
Extract frames from clips or full videos for embeddings, filtering, and analysis.
+++
{bdg-primary}`frames`
{bdg-secondary}`nvdec`
{bdg-secondary}`ffmpeg`
{bdg-secondary}`fps`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Create Embeddings
:link: video-process-embeddings
:link-type: ref
Generate clip-level embeddings with InternVideo2 or Cosmos-Embed1 for search and duplicate removal.
+++
{bdg-primary}`clips`
{bdg-secondary}`internvideo2`
{bdg-secondary}`cosmos-embed1`
:::

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Create Captions & Preview
:link: video-process-captions-preview
:link-type: ref
Generate Qwen‑VL captions and optional WebP previews; optionally enhance with Qwen‑LM.
+++
{bdg-primary}`captions`
{bdg-primary}`previews`
{bdg-secondary}`qwen`
{bdg-secondary}`webp`
:::

:::{grid-item-card} {octicon}`git-branch;1.5em;sd-mr-1` Remove Duplicate Embeddings
:link: video-process-dedup
:link-type: ref
Remove near-duplicates using semantic clustering and similarity with generated embeddings.
+++
{bdg-primary}`clips`
{bdg-secondary}`semantic`
{bdg-secondary}`pairwise`
{bdg-secondary}`kmeans`
:::

::::

### Write Data

Save outputs in formats your training or retrieval systems can consume at scale.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

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

Practice with guided, hands-on examples to build, customize, and run video pipelines.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Beginner Tutorial
:link: video-tutorials-beginner
:link-type: ref
Create and run your first video pipeline: read, split, encode, embed, write.
+++
{bdg-secondary}`splitting`
{bdg-secondary}`encoding`
{bdg-secondary}`embeddings`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Pipeline Customization Tutorials
:link: video-tutorials-pipeline-cust-series
:link-type: ref
Customize environments, code, models, and stages for video pipelines.
+++
{bdg-secondary}`environments`
{bdg-secondary}`code`
{bdg-secondary}`models`
{bdg-secondary}`stages`
:::

::::
