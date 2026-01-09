---
description: "Comprehensive text curation capabilities for preparing high-quality data for large language model training with loading, filtering, and quality assessment"
categories: ["workflows"]
tags: ["text-curation", "data-loading", "filtering", "deduplication", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "workflow"
modality: "text-only"
---

(text-overview)=
# About Text Curation

NeMo Curator provides comprehensive text curation capabilities to prepare high-quality data for large language model (LLM) training. The toolkit includes a collection of processors for loading, filtering, formatting, and analyzing text data from various sources using a {ref}`pipeline-based architecture <about-concepts-text-data-curation-pipeline>`.

## Use Cases

- Clean and prepare web-scraped data from sources like Common Crawl, Wikipedia, and arXiv
- Create custom text curation pipelines for specific domain needs
- Scale text processing across CPU and GPU clusters efficiently

## Architecture

The following diagram provides a high-level outline of NeMo Curator's text curation architecture.

```{mermaid}
flowchart LR
    A["Data Sources<br/>(Cloud, Local,<br/>Common Crawl, arXiv,<br/>Wikipedia)"] --> B["Data Acquisition<br/>& Loading"]
    B --> C["Content Processing<br/>& Cleaning"]
    C --> D["Quality Assessment<br/>& Filtering"]
    D --> E["Deduplication<br/>(Exact, Fuzzy,<br/>Semantic)"]
    E --> F["Curated Dataset<br/>(JSONL/Parquet)"]
    
    G["Ray + RAPIDS<br/>(GPU-accelerated)"] -.->|"Distributed Execution"| B
    G -.->|"Distributed Execution"| C
    G -.->|"GPU Acceleration"| D
    G -.->|"GPU Acceleration"| E

    classDef stage fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef infra fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000

    class A,B,C,D,E stage
    class F output
    class G infra
```

---

## Introduction

Master the fundamentals of NeMo Curator and set up your text processing environment.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Concepts
:link: about-concepts-text
:link-type: ref
Learn about pipeline architecture and core processing stages for efficient text curation
+++
{bdg-secondary}`data-structures`
{bdg-secondary}`distributed`
{bdg-secondary}`architecture`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: gs-text
:link-type: ref
Learn prerequisites, setup instructions, and initial configuration for text curation
+++
{bdg-secondary}`setup`
{bdg-secondary}`configuration`
{bdg-secondary}`quickstart`
:::

::::

## Curation Tasks

### Download Data

Download text data from remote sources and import existing datasets into NeMo Curator's processing pipeline.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`file;1.5em;sd-mr-1` Read Existing Data
:link: text-load-data-read-existing
:link-type: ref
Read existing JSONL and Parquet datasets using Curator's reader stages
+++
{bdg-secondary}`jsonl`
{bdg-secondary}`parquet`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` arXiv
:link: text-load-data-arxiv
:link-type: ref
Download and extract scientific papers from arXiv
+++
{bdg-secondary}`academic`
{bdg-secondary}`pdf`
{bdg-secondary}`latex`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Common Crawl
:link: text-load-data-common-crawl
:link-type: ref
Download and extract web archive data from Common Crawl
+++
{bdg-secondary}`web-data`
{bdg-secondary}`warc`
{bdg-secondary}`distributed`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Wikipedia
:link: text-load-data-wikipedia
:link-type: ref
Download and extract Wikipedia articles from Wikipedia dumps
+++
{bdg-secondary}`articles`
{bdg-secondary}`multilingual`
{bdg-secondary}`dumps`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Custom Data Sources
:link: text-load-data-custom
:link-type: ref
Implement a download and extract pipeline for a custom data source
+++
{bdg-secondary}`jsonl`
{bdg-secondary}`parquet`
{bdg-secondary}`custom-formats`
:::

::::

### Process Data

Transform and enhance your text data through comprehensive processing and curation steps.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Language Management
:link: process-data/language-management/index
:link-type: doc
Handle multilingual content and language-specific processing
+++
{bdg-secondary}`language-detection`
{bdg-secondary}`stopwords`
{bdg-secondary}`multilingual`
:::

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Content Processing & Cleaning
:link: process-data/content-processing/index
:link-type: doc
Clean, normalize, and transform text content
+++
{bdg-secondary}`cleaning`
{bdg-secondary}`normalization`
{bdg-secondary}`formatting`
:::

:::{grid-item-card} {octicon}`duplicate;1.5em;sd-mr-1` Deduplication
:link: process-data/deduplication/index
:link-type: doc
Remove duplicate and near-duplicate documents efficiently
+++
{bdg-secondary}`fuzzy-dedup`
{bdg-secondary}`semantic-dedup`
{bdg-secondary}`exact-dedup`
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Quality Assessment & Filtering
:link: process-data/quality-assessment/index
:link-type: doc
Score and remove low-quality content
+++
{bdg-secondary}`heuristics`
{bdg-secondary}`classifiers`
{bdg-secondary}`quality-scoring`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Specialized Processing
:link: process-data/specialized-processing/index
:link-type: doc
Domain-specific processing for code and advanced curation tasks
+++
{bdg-secondary}`code-processing`
:::

:::{grid-item-card} {octicon}`sparkles;1.5em;sd-mr-1` Synthetic Data Generation
:link: synthetic/index
:link-type: doc
Generate and augment training data using LLMs
+++
{bdg-secondary}`llm`
{bdg-secondary}`augmentation`
{bdg-secondary}`multilingual`
{bdg-secondary}`nemotron-cc`
:::

::::


<!-- ## Tutorials

Build practical experience with step-by-step guides for common text curation workflows.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Text Curation Tutorials (Placeholder)
:link: tutorials/index
:link-type: doc
Learn how to customize NeMo Curator's pipelines for your specific needs
+++
{bdg-primary}`staged-nolink`
{bdg-secondary}`custom-pipelines`
{bdg-secondary}`optimization`
{bdg-secondary}`examples`
:::

:::: -->
