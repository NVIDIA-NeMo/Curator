---
description: "Load image data for curation using WebDataset format with distributed processing and GPU acceleration"
categories: ["workflows"]
tags: ["data-loading", "webdataset", "distributed", "dali", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "image-only"
---

(image-load-data)=
# Image Data Loading

Load image data for curation using NeMo Curator. The primary supported format is WebDataset, which enables efficient distributed processing and annotation of large-scale image-text datasets.

## How it Works

NeMo Curator's image data loading is optimized for large-scale, distributed curation workflows:

1. **Sharded WebDataset Format**: Image, caption, and metadata files are grouped into sharded `.tar` archives, with corresponding `.parquet` files for fast metadata access.

2. **Unified Metadata**: Each record is uniquely identified and linked across image, caption, and metadata files, enabling efficient distributed processing.

3. **High-Performance Loading**: Optional `.idx` index files enable NVIDIA DALI to accelerate data loading, shuffling, and batching on GPU.

4. **Cloud and Local Storage**: Datasets can be loaded from local disk or cloud storage (S3, GCS, Azure) using the same API.

5. **Standardized Loader**: The `ImageTextPairDataset.from_webdataset` method loads the entire dataset structure in one step—no need for separate downloaders, iterators, or extractors.

The result is a standardized `ImageTextPairDataset` ready for embedding, classification, and filtering in downstream curation pipelines.

---

## Options

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` WebDataset
:link: image-load-data-webdataset
:link-type: ref
Load and process sharded image-text datasets in the WebDataset format for scalable distributed curation.
+++
{bdg-secondary}`webdataset`
{bdg-secondary}`sharded`
{bdg-secondary}`distributed`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Webdataset <webdataset>
```
