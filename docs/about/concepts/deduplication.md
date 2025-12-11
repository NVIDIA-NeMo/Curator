---
description: "Comprehensive overview of deduplication techniques across text, image, and video modalities including exact, fuzzy, and semantic approaches"
categories: ["concepts-architecture"]
tags: ["deduplication", "exact-dedup", "fuzzy-dedup", "semantic-dedup", "multimodal", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "multimodal"
---

(about-concepts-deduplication)=

# Deduplication Concepts

This guide covers deduplication techniques available across all modalities in NeMo Curator, from exact hash-based matching to semantic similarity detection using embeddings.

## Overview

Deduplication is a critical step in data curation that removes duplicate and near-duplicate content to improve model training efficiency. NeMo Curator provides sophisticated deduplication capabilities that work across text and image modalities.

Removing duplicates offers several benefits:

- **Improved Training Efficiency**: Prevents overrepresentation of repeated content
- **Reduced Dataset Size**: Significantly reduces storage and processing requirements
- **Better Model Performance**: Eliminates redundant examples that can bias training

## Deduplication Approaches

NeMo Curator implements three main deduplication strategies, each with different strengths and use cases:

### Exact Deduplication

- **Method**: Hash-based matching (MD5)
- **Best For**: Identical copies and character-for-character matches
- **Speed**: Very fast
- **Scale**: Unlimited size
- **GPU Required**: Yes (for distributed processing)

Exact deduplication identifies documents or media files that are completely identical by computing cryptographic hashes of their content.

**Modalities Supported**: Text

### Fuzzy Deduplication

- **Method**: MinHash and Locality-Sensitive Hashing (LSH)
- **Best For**: Near-duplicates with minor changes (reformatting, small edits)
- **Speed**: Fast
- **Scale**: Up to petabyte scale
- **GPU Required**: Yes

Fuzzy deduplication uses statistical fingerprinting to identify content that is nearly identical but may have small variations like formatting changes or minor edits.

**Modalities Supported**: Text

### Semantic Deduplication

- **Method**: Embedding-based similarity using neural networks
- **Best For**: Content with similar meaning but different expression
- **Speed**: Moderate (due to embedding generation step)
- **Scale**: Up to terabyte scale
- **GPU Required**: Yes

Semantic deduplication leverages deep learning embeddings to identify content that conveys similar meaning despite using different words, visual elements, or presentation.

**Modalities Supported**: Text, Image

## Multimodal Applications

### Text Deduplication

Text deduplication is the most mature implementation, offering all three approaches:

- **Exact**: Remove identical documents using MD5 hashing
- **Fuzzy**: Remove near-duplicates using MinHash and LSH similarity
- **Semantic**: Remove semantically similar content using embeddings

Text deduplication can handle web-scale datasets and is commonly used for:

- Web crawl data (Common Crawl)
- Academic papers (ArXiv)
- Code repositories
- General text corpora

### Video Deduplication

Video deduplication uses the semantic deduplication workflow with video embeddings:

- **Semantic Clustering**: Uses the general K-means clustering workflow on video embeddings
- **Pairwise Similarity**: Computes within-cluster similarity using the semantic deduplication pipeline
- **Representative Selection**: Leverages the semantic workflow to identify and remove redundant content

Video deduplication is particularly effective for:

- Educational content with similar presentations
- News clips covering the same events
- Entertainment content with repeated segments

### Image Deduplication

Semantic duplicates are images that contain almost the same information content, but are perceptually different.

Image deduplication is computed in Curator by:

- **Generating Embeddings**: Generate CLIP embeddings for images
- **Convert to Text**: Convert the `ImageBatch` embeddings to `DocumentBatch` objects
- **Identify Semantic Duplicates**: Run the text-based semantic deduplication workflow and save the results
- **Remove Duplicates**: Read back the data and remove the identified duplicates

## Architecture and Performance

### Distributed Processing

All deduplication workflows leverage distributed computing frameworks:

- **Ray Backend**: Provides scalable distributed processing
- **GPU Acceleration**: Essential for embedding generation and similarity computation
- **Memory Optimization**: Streaming processing for large datasets

### Scalability Characteristics

```{list-table} Deduplication Scalability
:header-rows: 1
:widths: 20 25 25 30

* - Method
  - Dataset Size
  - Memory Requirements
  - Processing Time
* - Exact
  - Unlimited
  - Low (hash storage)
  - Linear with data size
* - Fuzzy
  - Petabyte-scale
  - Moderate (LSH tables)
  - Sub-linear with LSH
* - Semantic
  - Terabyte-scale
  - High (embeddings)
  - Depends on model inference
```

## Implementation Patterns

### Workflow-Based Processing

NeMo Curator provides high-level workflows that encapsulate the complete deduplication process:

```python
# Text-based workflow for identifying exact duplicates
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

# Text-based workflow for identifying fuzzy duplicates
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

# Text-based workflow for identifying (and optionally removing) semantic duplicates
from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow

# Text-based workflow for removing identified duplicates
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
```

### Stage-Based Processing

For fine-grained control, individual stages can be composed into custom pipelines:

```python
# Semantic deduplication stages
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansStage
from nemo_curator.stages.deduplication.semantic.pairwise import PairwiseStage
from nemo_curator.stages.deduplication.semantic.identify_duplicates import IdentifyDuplicatesStage
```

## Integration with Pipeline Architecture

Deduplication workflows should be run separately from traditional pipelines in Curator. While other Curator modules are purely map-style operations, meaning that they run on chunks of the data at a time, deduplication workflows identify duplicates across the entire dataset. Thus, special logic is needed outside of Curator's map-style functions, and the deduplication modules are implemented as separate workflows.

Each deduplication workflow expects input and output file path parameters, making them more self-contained than other Curator modules. The input and output of a deduplication workflow can be JSONL or Parquet files, meaning that they are compatible with Curator's traditional pipeline-based read and write stages. Thus, the user may write out intermediate results of a pipeline to be used as input to a deduplication workflow, then read the deduplicated output and resume curation.

As a high level example:

```python
# Define first pipeline
pipeline_1 = Pipeline(name="text_curation_1", description="...")
# Read input data
pipeline_1.add_stage(JsonlReader(...))
# Add more stages like heuristic filters, etc.
pipeline_1.add_stage(...)
# Save intermediate results to JSONL
pipeline_1.add_stage(JsonlWriter(...))
pipeline_1.run()

# Create and run semantic deduplication workflow
workflow = SemanticDeduplicationWorkflow(...)
workflow.run()

# Define second pipeline
pipeline_2 = Pipeline(name="text_curation_2", description="...")
# Read deduplicated data
pipeline_2.add_stage(JsonlReader(...))
# Add more stages like classifiers, etc.
pipeline_2.add_stage(...)
# Save final results to JSONL
pipeline_2.add_stage(JsonlWriter(...))
pipeline_2.run()
```
