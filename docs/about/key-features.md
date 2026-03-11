---
description: "Comprehensive overview of NeMo Curator's key features for text, image, video, and audio data curation with deployment options"
categories: ["concepts-architecture"]
tags: ["features", "benchmarks", "deduplication", "classification", "gpu-accelerated", "distributed", "deployment-operations"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused", "devops-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(about-key-features)=

# Key Features

NeMo Curator is an enterprise-grade platform for scalable, privacy-aware data curation across text, image, video, and audio. It empowers teams to prepare high-quality, compliant datasets for LLM and AI training, with robust support for distributed, cloud-native, and on-premises workflows. Leading organizations trust NeMo Curator for its modular pipelines, advanced filtering, and seamless integration with modern MLOps environments.

## Why NeMo Curator?

- Trusted by leading organizations for LLM and generative AI data curation
- Open source, NVIDIA-supported, and actively maintained
- Seamless integration with enterprise MLOps and data platforms
- Proven at scale: from laptops to multi-node GPU clusters

### Benchmarks & Results

- **Deduplicated 1.96 trillion tokens in 0.5 hours** using 32 NVIDIA H100 GPUs (RedPajama V2 scale)
- Up to **80% data reduction** and significant improvements in downstream model performance (see ablation studies)
- Efficient curation of Common Crawl: from 2.8TB raw to 0.52TB high-quality data in under 38 hours on 30 CPU nodes

---

## Text Data Curation

NeMo Curator offers advanced tools for text data loading, cleaning, filtering, deduplication, and classification. Built-in modules support language identification, quality estimation, domain and safety classification. Pipelines are fully modular and can be customized for diverse NLP and LLM training needs.


::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data Loading
:link: about-concepts-text-data-loading
:link-type: ref
Efficiently load and manage massive text datasets, with support for common formats and scalable streaming.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Data Processing
:link: about-concepts-text-data-processing
:link-type: ref
Advanced filtering, deduplication, classification, and pipeline design for high-quality text curation.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Text Curation Quickstart
:link: gs-text
:link-type: ref
Set up your environment and run your first text curation pipeline with NeMo Curator.
:::

::::

---

## Image Data Curation

NeMo Curator supports scalable image dataset loading, embedding, classification (aesthetic, NSFW, etc.), filtering, deduplication, and export. It leverages state-of-the-art vision models (for example, CLIP, timm) with pipeline-based architecture for efficient GPU-accelerated processing. Modular pipelines enable rapid experimentation and integration with text and multimodal workflows.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data Loading
:link: about-concepts-image-data-loading
:link-type: ref
Load and manage large-scale image datasets for curation workflows.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Data Processing
:link: about-concepts-image-data-processing
:link-type: ref
Embedding generation, classification (aesthetic, NSFW), filtering, and deduplication for images.
:::

:::{grid-item-card} {octicon}`device-camera;1.5em;sd-mr-1` Data Export
:link: about-concepts-image-data-export
:link-type: ref
Export, save, and reshard curated image datasets for downstream use.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Image Curation Quickstart
:link: gs-image
:link-type: ref
Set up your environment and install NeMo Curator's image modules.
:::

::::

---

## Audio Data Curation

NeMo Curator provides speech and audio curation capabilities designed for preparing high-quality speech datasets for ASR model training and multimodal applications. It leverages pretrained `.nemo` model checkpoints via the NeMo Framework for transcription, quality assessment through Word Error Rate (WER) calculation, and seamless integration with text curation workflows.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data Loading
:link: about-concepts-audio-manifests-ingest
:link-type: ref
Load and manage audio datasets with manifests, file paths, and transcriptions for curation workflows.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` ASR Processing
:link: about-concepts-audio-asr-pipeline
:link-type: ref
Automatic speech recognition inference, quality assessment, and transcription using NeMo Framework models.
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Quality Assessment
:link: about-concepts-audio-quality-metrics
:link-type: ref
Word Error Rate (WER) calculation, duration analysis, and quality-based filtering for speech data.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Audio Curation Quickstart
:link: gs-audio
:link-type: ref
Set up your environment and run your first audio curation pipeline with NeMo Curator.
:::

::::

---

## Video Data Curation

NeMo Curator provides distributed video curation pipelines, supporting scalable data flow, pipeline stages, and efficient processing for large video corpora. The architecture supports high-throughput, cloud-native, and on-prem deployments.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Architecture
:link: about-concepts-video-architecture
:link-type: ref

Distributed processing, Ray-based foundation, and autoscaling for video curation.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Key Abstractions
:link: about-concepts-video-abstractions
:link-type: ref

Stages, pipelines, and execution modes in video curation workflows.
:::

:::{grid-item-card} {octicon}`sync;1.5em;sd-mr-1` Data Flow
:link: about-concepts-video-data-flow
:link-type: ref

How data moves through the system, from ingestion to output, for efficient large-scale video curation.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Video Curation Quickstart
:link: gs-video
:link-type: ref
Set up your environment and run your first video curation pipeline with NeMo Curator.
:::

::::

## Deployment and Integration

NeMo Curator is designed for distributed, cloud-native, and on-premises deployments. It integrates easily with your existing MLOps pipelines. Modular APIs enable flexible orchestration and automation.


::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Deployment Options
:link: admin-overview
:link-type: ref
See the Admin Guide for deployment guidance and infrastructure recommendations.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Memory Management
:link: reference-infra-memory-management
:link-type: ref
Optimize memory usage and partitioning for large-scale curation workflows.
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` GPU Acceleration
:link: reference-infra-gpu-processing
:link-type: ref
Leverage NVIDIA GPUs for faster data processing and pipeline acceleration.
:::

:::{grid-item-card} {octicon}`sync;1.5em;sd-mr-1` Resumable Processing
:link: reference-infra-resumable-processing
:link-type: ref
Continue interrupted operations and recover large dataset processing with checkpointing and batching.
:::

::::
