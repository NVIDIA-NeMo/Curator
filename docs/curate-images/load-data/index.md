---
description: "Load image data for curation using tar archives with distributed processing and GPU acceleration"
categories: ["workflows"]
tags: ["data-loading", "tar-archives", "distributed", "dali", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "image-only"
---

(image-load-data)=
# Image Data Loading

Load image data for curation using NeMo Curator. The primary supported format is tar archives containing JPEG images, which enables efficient distributed processing of large-scale image datasets.

## How it Works

NeMo Curator's image data loading uses a pipeline-based approach optimized for large-scale, distributed curation workflows:

1. **File Partitioning**: `FilePartitioningStage` distributes `.tar` files across workers for parallel processing.

2. **High-Performance Reading**: `ImageReaderStage` uses NVIDIA DALI to accelerate image loading, decoding, and batching on GPU with CPU fallback.

3. **Tar Archive Format**: Processes sharded `.tar` archives containing JPEG images (other file types are ignored during loading).

4. **Batch Processing**: Images are processed in `ImageBatch` objects containing decoded images, metadata, and processing results.

The result is a stream of `ImageBatch` objects ready for embedding, classification, and filtering in downstream pipeline stages.

---

## Options

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Tar Archive Pipeline
:link: image-load-data-tar-archives
:link-type: ref
Load and process JPEG images from tar archives using `FilePartitioningStage` and `ImageReaderStage` for scalable distributed curation.
+++
{bdg-secondary}`FilePartitioningStage`
{bdg-secondary}`ImageReaderStage`
{bdg-secondary}`DALI-accelerated`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Tar Archives <tar-archives>
```
