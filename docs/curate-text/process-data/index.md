---
description: "Process text data using comprehensive filtering, deduplication, content processing, and specialized tools for high-quality datasets"
categories: ["workflows"]
tags: ["data-processing", "filtering", "deduplication", "content-processing", "quality-assessment", "distributed"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

# Process Data for Text Curation

Process text data you've loaded through NeMo Curator's {ref}`pipeline architecture <about-concepts-text-data-loading>`.

NeMo Curator provides a comprehensive suite of tools for processing text data as part of the AI training pipeline. These tools help you analyze, transform, and filter your text datasets to ensure high-quality input for language model training.

## How it Works

NeMo Curator's text processing capabilities are organized into five main categories:

1. **Quality Assessment & Filtering**: Score and remove low-quality content using heuristics and ML classifiers
2. **Deduplication**: Remove duplicate and near-duplicate documents efficiently
3. **Content Processing & Cleaning**: Clean, normalize, and transform text content
4. **Language Management**: Handle multilingual content and language-specific processing
5. **Specialized Processing**: Domain-specific processing for code and advanced curation tasks

Each category provides specific implementations optimized for different curation needs. The result is a cleaned and filtered dataset ready for model training.

---

## Quality Assessment & Filtering

Score and remove low-quality content using heuristics and ML classifiers.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Heuristic Filtering
:link: quality-assessment/heuristic
:link-type: doc
Filter text using configurable rules and metrics
+++
{bdg-secondary}`rules`
{bdg-secondary}`metrics`
{bdg-secondary}`fast`
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Classifier Filtering
:link: quality-assessment/classifier
:link-type: doc
Filter text using trained quality classifiers
+++
{bdg-secondary}`ml-models`
{bdg-secondary}`quality`
{bdg-secondary}`scoring`
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Distributed Classification
:link: quality-assessment/distributed-classifier
:link-type: doc
GPU-accelerated classification with pre-trained models
+++
{bdg-secondary}`gpu`
{bdg-secondary}`distributed`
{bdg-secondary}`scalable`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Custom Filters
:link: quality-assessment/custom
:link-type: doc
Implement and combine your own custom filters
+++
{bdg-secondary}`custom`
{bdg-secondary}`flexible`
{bdg-secondary}`extensible`
:::

::::

## Deduplication

Remove duplicate and near-duplicate documents efficiently from your text datasets.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`git-pull-request;1.5em;sd-mr-1` Exact Duplicate Removal
:link: deduplication/exact
:link-type: doc
Identify character-for-character duplicates using hashing
+++
{bdg-secondary}`hashing`
{bdg-secondary}`fast`
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` Fuzzy Duplicate Removal
:link: deduplication/fuzzy
:link-type: doc
Identify near-duplicates using MinHash and LSH
+++
{bdg-secondary}`minhash`
{bdg-secondary}`lsh`
{bdg-secondary}`gpu-accelerated`
:::

:::{grid-item-card} {octicon}`repo-clone;1.5em;sd-mr-1` Semantic Deduplication
:link: deduplication/semdedup
:link-type: doc
Remove semantically similar documents using embeddings
+++
{bdg-secondary}`embeddings`
{bdg-secondary}`meaning-based`
{bdg-secondary}`gpu-accelerated`
:::

::::

## Content Processing & Cleaning

Clean, normalize, and transform text content for high-quality training data.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`shield-lock;1.5em;sd-mr-1` PII Removal
:link: content-processing/pii
:link-type: doc
Identify and remove personal identifiable information
+++
{bdg-secondary}`privacy`
{bdg-secondary}`masking`
{bdg-secondary}`compliance`
:::

:::{grid-item-card} {octicon}`typography;1.5em;sd-mr-1` Text Cleaning
:link: content-processing/text-cleaning
:link-type: doc
Fix Unicode issues, standardize spacing, and remove URLs
+++
{bdg-secondary}`unicode`
{bdg-secondary}`normalization`
{bdg-secondary}`preprocessing`
:::

::::

## Language Management

Handle multilingual content and language-specific processing requirements.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Language Identification
:link: language-management/language
:link-type: doc
Identify document languages and separate multilingual datasets
+++
{bdg-secondary}`fasttext`
{bdg-secondary}`176-languages`
{bdg-secondary}`detection`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Stop Words
:link: language-management/stopwords
:link-type: doc
Manage high-frequency words to enhance text extraction and content detection
+++
{bdg-secondary}`preprocessing`
{bdg-secondary}`filtering`
{bdg-secondary}`language-specific`
:::

::::

## Specialized Processing

Domain-specific processing for code and advanced curation tasks.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Processing
:link: specialized-processing/code
:link-type: doc
Specialized filters for programming content and source code
+++
{bdg-secondary}`programming`
{bdg-secondary}`syntax`
{bdg-secondary}`comments`
:::

::::

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Quality Assessment & Filtering <quality-assessment/index>
Deduplication <deduplication/index>
Content Processing & Cleaning <content-processing/index>
Language Management <language-management/index>
Specialized Processing <specialized-processing/index>
```
