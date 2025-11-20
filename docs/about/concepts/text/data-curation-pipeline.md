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

- **Cloud storage**: S3, GCS, Azure (via `s3://`, `gcs://`, `az://` protocols)
- **Local workstation**: JSONL and Parquet files

### 2. Data Acquisition & Processing

Raw data is downloaded, extracted, and converted into standardized formats:

- **Download & Extraction**: Retrieve and process remote data sources using source-specific downloaders
- **Cleaning & Pre-processing**: Convert formats and normalize text during extraction
- **DocumentBatch Creation**: Standardize data into `DocumentBatch`, NeMo Curator's core data structure that wraps PyArrow tables or Pandas DataFrames

### 3. Quality Assessment & Filtering

Multiple filtering stages ensure data quality:

- **Heuristic Quality Filtering**: Rule-based filters such as token count, substring matching, and pattern detection
- **Model-based Quality Filtering**: ML classification models (like FastText) trained to score and identify high-quality text

### 4. Deduplication

Remove duplicate and near-duplicate content at scale:

- **Exact Deduplication**: Identify identical documents using MD5 hashing for fast exact matching
- **Fuzzy Deduplication**: Detect near-duplicates using MinHash signatures with Locality-Sensitive Hashing (LSH) for efficient similarity search
- **Semantic Deduplication**: Find semantically similar content using text embeddings with K-means clustering and pairwise similarity comparison

### 5. Final Preparation

Prepare the curated dataset for training:

- **Format Standardization**: Ensure consistent output format

## Infrastructure Foundation

The entire pipeline runs on a robust, scalable infrastructure:

- **Ray**: Primary distributed computing framework for parallelization across nodes
- **RAPIDS**: GPU-accelerated data processing libraries (cuDF for DataFrames, cuGraph for graph operations, cuML for machine learning)
- **Flexible Deployment**: CPU-only and GPU-accelerated modes supported

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
- **Multi-Node**: Distribute processing across Ray clusters with multiple worker nodes
- **Cloud Native**: Deploy on cloud platforms (AWS, GCP, Azure) with cloud storage integration
- **On-Premises Clusters**: Run on GPU clusters and data center infrastructure

---

For hands-on experience, see the {ref}`Text Curation Getting Started Guide <gs-text>`.