---
description: "Overview of NeMo Curator, an open-source platform for scalable data curation across text, image, video, and audio modalities for AI training"
categories: ["getting-started"]
tags: ["overview", "platform", "multimodal", "enterprise", "getting-started"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused", "devops-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(about-overview)=

# Overview of NeMo Curator

NeMo Curator is an open-source, enterprise-grade platform for scalable, privacy-aware data curation across text, image, video, and audio modalities.

NeMo Curator, part of the NVIDIA NeMo software suite for managing the AI agent lifecycle, helps you prepare high-quality, compliant datasets for large language model (LLM) and generative artificial intelligence (AI) training. Whether you work in the cloud, on-premises, or in a hybrid environment, NeMo Curator supports your workflow.

## Target Users

- **Data scientists and machine learning engineers**: Build and curate datasets for LLMs, generative models, and multimodal AI.
- **Cluster administrators and DevOps professionals**: Deploy and scale curation pipelines on Kubernetes, Slurm, or Apache Spark clusters.
- **Researchers**: Experiment with new data curation techniques and ablation studies.
- **Enterprises**: Ensure data privacy, compliance, and quality for production AI workflows.

## How It Works

NeMo Curator speeds up data curation by using modern hardware and distributed computing frameworks. You can process data efficiently—from a single laptop to a multi-node GPU cluster. With modular pipelines, advanced filtering, and easy integration with machine learning operations (MLOps) tools, leading organizations trust NeMo Curator.

- **Text Curation**: Uses a pipeline-based architecture with modular processing stages running on Ray. Data flows through data download, extraction, language detection, rule-based quality filtering, deduplication (exact, fuzzy and semantic) and model based quality filtering.
- **Image Curation**: Uses pipeline-based architecture with modular stages for loading, embedding generation, classification (aesthetic, NSFW), filtering, and export workflows. Supports distributed processing with optional GPU acceleration.
- **Video Curation**: Employs Ray-based pipelines to split long videos into clips using fixed stride or scene-change detection, with optional encoding, filtering, embedding generation, and deduplication for large-scale video processing.
- **Audio Curation**: Provides ASR inference using models, quality assessment through Word Error Rate (WER) calculation, duration analysis, and integration with text curation workflows for speech data processing.

### Key Technologies

- **Graphics Processing Units (GPUs)**: Speed up data processing for large-scale workloads.
- **Distributed Computing**: Supports frameworks like Dask, RAPIDS, and Ray for scalable, parallel processing.
- **Modular Pipelines**: Build, customize, and scale curation workflows to fit your needs.
- **MLOps Integration**: Seamlessly connects with modern MLOps environments for production-ready workflows.

## Concepts

Explore the foundational concepts and terminology used across NeMo Curator.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`typography;1.5em;sd-mr-1` Text Curation Concepts
:link: about-concepts-text
:link-type: ref

Learn about text data curation, covering data loading and processing (filtering, deduplication, classification).
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Image Curation Concepts
:link: about-concepts-image
:link-type: ref

Explore key concepts for image data curation, including scalable loading, processing (embedding, classification, filtering, deduplication), and dataset export.
:::

:::{grid-item-card} {octicon}`video;1.5em;sd-mr-1` Video Curation Concepts
:link: about-concepts-video
:link-type: ref

Discover video data curation concepts, such as distributed processing, pipeline stages, execution modes, and efficient data flow.
:::

:::{grid-item-card} {octicon}`unmute;1.5em;sd-mr-1` Audio Curation Concepts
:link: about-concepts-audio
:link-type: ref

Learn about speech data curation, ASR inference, quality assessment, and audio-text integration workflows.
:::

::::
