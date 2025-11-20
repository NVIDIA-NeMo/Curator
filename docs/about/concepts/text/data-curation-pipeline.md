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

```{mermaid}
flowchart TB
    subgraph Sources["Data Sources"]
        S1["Cloud Storage<br/>(S3, Azure)"]
        S2["Local Files<br/>(JSONL, Parquet)"]
        S3["Common Crawl<br/>(WARC)"]
        S4["arXiv<br/>(PDF, LaTeX)"]
        S5["Wikipedia<br/>(Dumps)"]
    end

    subgraph Acquisition["Data Acquisition & Loading"]
        A1["Download & Extract"]
        A2["JSONL/Parquet Reader"]
        A3["DocumentBatch Creation"]
    end

    subgraph Processing["Content Processing"]
        P1["Text Cleaning<br/>& Normalization"]
        P2["Language Detection<br/>& Management"]
        P3["Code Processing<br/>(Optional)"]
    end

    subgraph Quality["Quality Assessment & Filtering"]
        Q1["Heuristic Filtering<br/>(Rules-based)"]
        Q2["Classifier Filtering<br/>(ML Models)"]
        Q3["Distributed Classification<br/>(GPU-accelerated)"]
    end

    subgraph Dedup["Deduplication"]
        D1["Exact Deduplication<br/>(MD5 Hashing)"]
        D2["Fuzzy Deduplication<br/>(MinHash/LSH)"]
        D3["Semantic Deduplication<br/>(Embeddings)"]
    end

    subgraph Output["Final Output"]
        O1["Format Standardization"]
        O2["Curated Dataset<br/>(JSONL/Parquet)"]
    end

    subgraph Infrastructure["Infrastructure Layer"]
        I1["Ray<br/>(Distributed Computing)"]
        I2["RAPIDS<br/>(cuDF, cuGraph, cuML)"]
        I3["XennaExecutor<br/>(Auto-scaling)"]
    end

    Sources --> Acquisition
    Acquisition --> Processing
    Processing --> Quality
    Quality --> Dedup
    Dedup --> Output

    Infrastructure -.->|"Execution Backend"| Acquisition
    Infrastructure -.->|"GPU Acceleration"| Processing
    Infrastructure -.->|"GPU Acceleration"| Quality
    Infrastructure -.->|"GPU Acceleration"| Dedup

    classDef source fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#000
    classDef process fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef quality fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
    classDef dedup fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef output fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px,color:#000
    classDef infra fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000

    class S1,S2,S3,S4,S5 source
    class A1,A2,A3,P1,P2,P3 process
    class Q1,Q2,Q3 quality
    class D1,D2,D3 dedup
    class O1,O2 output
    class I1,I2,I3 infra
```

:::{note}
This diagram reflects the current 25.09 architecture using Ray as the distributed computing framework. Features such as PII removal, synthetic data generation, task decontamination, and blending/shuffling are not yet available but are planned for future releases. For details about current feature availability and limitations, refer to the Known Limitations section in the {ref}`release notes <about-release-notes>`.
:::

## Pipeline Stages

NeMo Curator's text curation pipeline consists of several key stages that work together to transform raw data sources into high-quality datasets ready for LLM training:

### 1. Data Sources

Multiple input sources provide the foundation for text curation:

- **Cloud storage**: Amazon S3, Azure
- **Local workstation**: JSONL, Parquet

### 2. Data Acquisition & Processing

Raw data is downloaded, extracted, and converted into standardized formats:

- **Download & Extraction**: Retrieve and process remote data sources
- **Cleaning & Pre-processing**: Convert formats and normalize text
- **DocumentBatch Creation**: Standardize data into NeMo Curator's core data structure

### 3. Quality Assessment & Filtering

Multiple filtering stages ensure data quality:

- **Heuristic Quality Filtering**: Rule-based filters for basic quality checks
- **Model-based Quality Filtering**: Classification models trained to identify high vs. low quality text

### 4. Deduplication

Remove duplicate and near-duplicate content:

- **Exact Deduplication**: Remove identical documents using MD5 hashing
- **Fuzzy Deduplication**: Remove near-duplicates using MinHash and LSH similarity
- **Semantic Deduplication**: Remove semantically similar content using embeddings

### 5. Final Preparation

Prepare the curated dataset for training:

- **Format Standardization**: Ensure consistent output format

## Infrastructure Foundation

The entire pipeline runs on a robust, scalable infrastructure:

- **Ray**: Distributed computing framework for parallelization
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
- **Cloud Native**: Deploy on cloud platforms
- **HPC Integration**: Run on HPC supercomputing clusters

---

For hands-on experience, refer to the {ref}`Text Curation Getting Started Guide <gs-text>`.