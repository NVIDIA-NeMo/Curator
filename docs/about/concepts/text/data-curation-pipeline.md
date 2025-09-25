---
description: "Comprehensive overview of NeMo Curator's text curation pipeline architecture including data acquisition and processing"
categories: ["concepts-architecture"]
tags: ["pipeline", "architecture", "text-curation", "distributed", "gpu-accelerated", "overview"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "text-only"
---

(about-concepts-text-data-curation-pipeline)=
# Text Data Curation Pipeline

This guide provides a comprehensive overview of NeMo Curator's text curation pipeline architecture, from data acquisition through final dataset preparation.

## Architecture Overview

The following diagram provides a high-level outline of NeMo Curator's text curation architecture:

```{image} _images/text-processing-diagram.png
:alt: High-level outline of NeMo Curator's text curation architecture
```

## Pipeline Stages

NeMo Curator's text curation pipeline consists of several key stages that work together to transform raw data sources into high-quality datasets ready for LLM training:

### 1. Data Sources
Multiple input sources provide the foundation for text curation:
- **Cloud storage** (S3, GCS, Azure)
- **Internet sources** (Common Crawl, ArXiv, Wikipedia)
- **Local workstation** files

### 2. Data Acquisition & Processing
Raw data is downloaded, extracted, and converted into standardized formats:
- **Download & Extraction**: Retrieve and process remote data sources
- **Cleaning & Pre-processing**: Convert formats and normalize text
- **DocumentBatch Creation**: Standardize data into NeMo Curator's core data structure

### 3. Quality Assessment & Filtering
Multiple filtering stages ensure data quality:
- **Heuristic Quality Filtering**: Rule-based filters for basic quality checks
- **Model-based Quality Filtering**: AI-powered content assessment
- **PII Removal**: Privacy-preserving data cleaning

### 4. Deduplication
Remove duplicate and near-duplicate content:
- **Exact Deduplication**: Remove identical documents using MD5 hashing
- **Fuzzy Deduplication**: Remove near-duplicates using MinHash and LSH similarity
- **Semantic Deduplication**: Remove semantically similar content using embeddings

### 5. Final Preparation
Prepare the curated dataset for training:
- **Blending/Shuffling**: Combine and randomize data sources
- **Format Standardization**: Ensure consistent output format

## Infrastructure Foundation

The entire pipeline runs on a robust, scalable infrastructure:
- **Dask**: Distributed computing framework for parallelization
- **RAPIDS**: GPU-accelerated data processing (cuDF, cuGraph, cuML)
- **Flexible Deployment**: CPU and GPU acceleration support

## Key Components

The pipeline leverages several core component types:

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data Loading
:link: about-concepts-text-data-loading
:link-type: ref

Core concepts for loading and managing text datasets from local files
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Data Acquisition
:link: about-concepts-text-data-acquisition
:link-type: ref

Components for downloading and extracting data from remote sources
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Data Processing
:link: about-concepts-text-data-processing
:link-type: ref

Concepts for filtering, deduplication, and classification
:::

::::

## Processing Modes

The pipeline supports different processing approaches:

**GPU Acceleration**: Leverage NVIDIA GPUs for:
- High-throughput data processing
- ML model inference for classification
- Embedding generation for semantic operations

**CPU Processing**: Scale across multiple CPU cores for:
- Text parsing and cleaning
- Rule-based filtering
- Large-scale data transformations

**Hybrid Workflows**: Combine CPU and GPU processing for optimal performance based on the specific operation.

## Scalability & Deployment

The architecture scales from single machines to large clusters:

- **Single Node**: Process datasets on laptops or workstations
- **Multi-Node**: Distribute processing across cluster resources
- **Cloud Native**: Deploy on Kubernetes or cloud platforms
- **HPC Integration**: Run on Slurm-managed supercomputing clusters

---

For hands-on experience, see the {ref}`Text Curation Getting Started Guide <gs-text>`.