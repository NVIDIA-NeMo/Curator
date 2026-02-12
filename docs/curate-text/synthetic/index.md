---
description: "Generate and augment training data using LLMs with NeMo Curator's synthetic data generation pipeline"
categories: ["workflows"]
tags: ["synthetic-data", "llm", "generation", "augmentation", "multilingual"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(synthetic-data-overview)=

# Synthetic Data Generation

NeMo Curator provides synthetic data generation (SDG) capabilities for creating and augmenting training data using Large Language Models (LLMs). These pipelines integrate with OpenAI-compatible APIs, enabling you to use NVIDIA NIM endpoints, local vLLM servers, or other inference providers.

## Use Cases

- **Data Augmentation**: Expand limited datasets by generating diverse variations
- **Multilingual Generation**: Create Q&A pairs and text in multiple languages
- **Knowledge Extraction**: Convert raw text into structured knowledge formats
- **Quality Improvement**: Paraphrase low-quality text into higher-quality Wikipedia-style prose
- **Training Data Creation**: Generate instruction-following data for model fine-tuning

## Core Concepts

Synthetic data generation in NeMo Curator operates in two primary modes:

### Generation Mode

Create new data from scratch without requiring input documents. The `QAMultilingualSyntheticStage` demonstrates this pattern—it generates Q&A pairs based on a prompt template without needing seed documents.

### Transformation Mode

Improve or restructure existing data using LLM capabilities. The Nemotron-CC stages exemplify this approach, taking input documents and producing:

- Paraphrased text in Wikipedia style
- Diverse Q&A pairs derived from document content
- Condensed knowledge distillations
- Extracted factual content

## Architecture

The following diagram shows how SDG pipelines process data through preprocessing, LLM generation, and postprocessing stages:

```{mermaid}
flowchart LR
    A["Input Documents<br/>(Parquet/JSONL)"] --> B["Preprocessing<br/>(Tokenization,<br/>Segmentation)"]
    B --> C["LLM Generation<br/>(OpenAI-compatible)"]
    C --> D["Postprocessing<br/>(Cleanup, Filtering)"]
    D --> E["Output Dataset<br/>(Parquet/JSONL)"]
    
    F["LLM Client<br/>(NVIDIA API,<br/>vLLM, TGI)"] -.->|"API Calls"| C
    
    classDef stage fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef infra fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000
    
    class A,B,C,D stage
    class E output
    class F infra
```

## Prerequisites

Before using synthetic data generation, ensure you have:

1. **NVIDIA API Key** (for cloud endpoints)
   - Obtain from [NVIDIA Build](https://build.nvidia.com/settings/api-keys)
   - Set as environment variable: `export NVIDIA_API_KEY="your-key"`

2. **NeMo Curator with text extras**

   ```bash
   uv pip install --extra-index-url https://pypi.nvidia.com nemo-curator[text_cuda12]
   ```

   :::{note}
   Nemotron-CC pipelines use the `transformers` library for tokenization, which is included in NeMo Curator's core dependencies.
   :::

## Available SDG Stages

```{list-table} Synthetic Data Generation Stages
:header-rows: 1
:widths: 30 40 30

* - Stage
  - Purpose
  - Input Type
* - `QAMultilingualSyntheticStage`
  - Generate multilingual Q&A pairs
  - Empty (generates from scratch)
* - `WikipediaParaphrasingStage`
  - Rewrite text as Wikipedia-style prose
  - Document text
* - `DiverseQAStage`
  - Generate diverse Q&A pairs from documents
  - Document text
* - `DistillStage`
  - Create condensed, information-dense paraphrases
  - Document text
* - `ExtractKnowledgeStage`
  - Extract knowledge as textbook-style passages
  - Document text
* - `KnowledgeListStage`
  - Extract structured fact lists
  - Document text
```

---

## Topics

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` LLM Client Setup
:link: llm-client
:link-type: doc
Configure OpenAI-compatible clients for NVIDIA APIs and custom endpoints
+++
{bdg-secondary}`configuration`
{bdg-secondary}`performance`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Multilingual Q&A Generation
:link: multilingual-qa
:link-type: doc
Generate synthetic Q&A pairs across multiple languages
+++
{bdg-secondary}`quickstart`
{bdg-secondary}`tutorial`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Nemotron-CC Pipelines
:link: nemotron-cc/index
:link-type: doc
Advanced text transformation and knowledge extraction workflows
+++
{bdg-secondary}`advanced`
{bdg-secondary}`paraphrasing`
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

llm-client
multilingual-qa
nemotron-cc/index
```
