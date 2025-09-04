---
description: "Overview of NeMo Curator, an open-source platform for scalable data curation across text, image, and video modalities for AI training"
categories: ["getting-started"]
tags: ["overview", "platform", "multimodal", "enterprise", "getting-started"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused", "devops-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(about-overview)=
# Overview of NeMo Curator

NeMo Curator is an open-source, enterprise-grade platform for scalable, privacy-aware data curation across text, image, and video modalities.

NeMo Curator, part of the NVIDIA NeMo software suite for managing the AI agent lifecycle, helps you prepare high-quality, compliant datasets for large language model (LLM) and generative artificial intelligence (AI) training. Whether you work in the cloud, on-premises, or in a hybrid environment, NeMo Curator supports your workflow.

## Target Users

- **Data scientists and machine learning engineers**: Build and curate datasets for LLMs, generative models, and multimodal AI.
- **Cluster administrators and DevOps professionals**: Deploy and scale curation pipelines on Kubernetes, Slurm, or Apache Spark clusters.
- **Researchers**: Experiment with new data curation techniques, synthetic data generation, and ablation studies.
- **Enterprises**: Ensure data privacy, compliance, and quality for production AI workflows.

## How It Works

NeMo Curator speeds up data curation by using modern hardware and distributed computing frameworks. You can process data efficiently—from a single laptop to a multi-node GPU cluster. With modular pipelines, advanced filtering, and easy integration with machine learning operations (MLOps) tools, NeMo Curator is trusted by leading organizations.

- **Text Curation**: Data flows through loaders and processors (cleaning, filtering, deduplication), and exporters, all built atop Dask for distributed execution.
- **Image Curation**: Uses WebDataset sharding, NVIDIA Data Loading Library (DALI) for GPU-accelerated loading, and modular steps for embedding, classification, filtering, and export.

### Key Technologies

- **Graphics Processing Units (GPUs)**: Accelerate data processing for large-scale workloads.
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

Learn about text data curation, covering data loading, processing (filtering, deduplication, classification), and synthetic data generation.
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

::::
