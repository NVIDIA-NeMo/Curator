---
description: "Overview of image data curation with NeMo Curator including loading, processing, filtering, and export workflows"
categories: ["workflows"]
tags: ["image-curation", "tar-archives", "filtering", "embedding", "workflows"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "workflow"
modality: "image-only"
---

(image-overview)=

# About Image Curation

Learn how to curate high-quality image datasets using NeMo Curator's powerful image processing pipeline. NeMo Curator enables you to efficiently process large-scale image-text datasets, applying quality filtering, content filtering, and semantic deduplication at scale.

## Use Cases

- Prepare high-quality image datasets for training generative AI models such as LLMs, VLMs, and WFMs
- Curate datasets for text-to-image model training and fine-tuning
- Process large-scale image collections for multimodal foundation model pretraining
- Apply quality control and content filtering to remove inappropriate or low-quality images
- Generate embeddings and semantic features for image search and retrieval applications
- Remove duplicate images from large datasets using semantic deduplication

## Architecture

NeMo Curator's image curation follows a modular pipeline architecture where data flows through configurable stages. Each stage performs a specific operation and passes processed data to the next stage in the pipeline.

```{mermaid}
flowchart LR
    A[Tar Archive Input] --> B[File Partitioning]
    B --> C[Image Reader<br/>DALI GPU-accelerated]
    C --> D[CLIP Embeddings<br/>ViT-L/14]
    D --> E[Aesthetic Filtering<br/>Quality scoring]
    E --> F[NSFW Filtering<br/>Content filtering]
    F --> G[Duplicate Removal<br/>Semantic deduplication]
    G --> H[Export & Sharding<br/>Tar + Parquet output]
    
    classDef input fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,color:#000
    classDef output fill:#e8f5e8,stroke:#2e7d32,color:#000
    
    class A input
    class B,C,D,E,F,G processing
    class H output
```

This pipeline architecture provides:

- **Modularity**: Add, remove, or reorder stages based on your workflow needs
- **Scalability**: Distributed processing across multiple GPUs and nodes using Ray
- **Flexibility**: Configure parameters for each stage independently
- **Efficiency**: GPU-accelerated processing with DALI and CLIP models

## Introduction

Master the fundamentals of NeMo Curator's image curation pipeline and set up your processing environment.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Concepts
:link: about-concepts-image
:link-type: ref
Learn about ImageBatch, ImageObject, and pipeline stages for efficient image curation
+++
{bdg-secondary}`data-structures`
{bdg-secondary}`distributed`
{bdg-secondary}`architecture`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: gs-image
:link-type: ref
Learn prerequisites, setup instructions, and initial configuration for image curation
+++
{bdg-secondary}`setup`
{bdg-secondary}`configuration`
{bdg-secondary}`quickstart`
:::

::::

## Curation Tasks

### Load Data

Load and process large-scale image datasets from local storage using tar archives with GPU-accelerated DALI for efficient distributed processing.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Tar Archives
:link: image-load-data-tar-archives
:link-type: ref
Load and process JPEG images from tar archives using DALI
+++
{bdg-secondary}`tar-archives`
{bdg-secondary}`dali`
{bdg-secondary}`gpu-accelerated`
:::

::::

### Process Data

Transform and enhance your image data through embeddings, classification, and filters.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Embeddings
:link: image-process-data-embeddings
:link-type: ref

Generate image embeddings using CLIP models.
+++
{bdg-secondary}`embeddings`

:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Filters
:link: image-process-data-filters
:link-type: ref

Apply built-in filters for aesthetic quality and NSFW content filtering.
+++
{bdg-secondary}`Aesthetic` {bdg-secondary}`NSFW` {bdg-secondary}`quality filtering`

:::

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Deduplication
:link: image-tutorials-dedup
:link-type: ref

Remove duplicate images using semantic similarity and clustering.
+++
{bdg-secondary}`deduplication` {bdg-secondary}`semantic` {bdg-secondary}`clustering`

:::

::::

### Pipeline Management

Optimize and manage your image curation pipelines with advanced execution backends and resource management.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Execution Backends
:link: reference-execution-backends
:link-type: ref

Configure Ray-based executors for distributed processing and resource management.
+++
{bdg-secondary}`ray` {bdg-secondary}`distributed` {bdg-secondary}`resource-management`

:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Performance Optimization
:link: image-load-data-tar-archives
:link-type: ref

Optimize performance with DALI GPU acceleration and efficient resource allocation.
+++
{bdg-secondary}`dali` {bdg-secondary}`gpu-acceleration` {bdg-secondary}`performance`

:::

::::

### Save & Export

Export your curated image datasets with metadata preservation, custom resharding options, and support for downstream training pipelines.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`device-camera;1.5em;sd-mr-1` Save & Export
:link: image-save-export
:link-type: ref

Save metadata to Parquet and export filtered datasets with custom resharding.
+++
{bdg-secondary}`parquet` {bdg-secondary}`tar-archives` {bdg-secondary}`resharding`

:::

::::
