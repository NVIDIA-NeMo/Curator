---
description: "Text processing workflows including quality filtering, fuzzy deduplication, content cleaning, and pipeline design"
categories: ["concepts-architecture"]
tags: ["data-processing", "quality-filtering", "deduplication", "pipeline", "pii-removal", "distributed"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "text-only"
---

(about-concepts-text-data-processing)=
# Text Processing Concepts

This guide covers the most common text processing workflows in NVIDIA NeMo Curator, based on real-world usage patterns from production data curation pipelines.

## Most Common Workflows

The majority of NeMo Curator users follow these core workflows, typically in this order:

### 1. Quality Filtering

Most users start with basic quality filtering using heuristic filters to remove low-quality content:

**Essential Quality Filters:**

- `WordCountFilter` - Remove too short/long documents
- `NonAlphaNumericFilter` - Remove symbol-heavy content
- `RepeatedLinesFilter` - Remove repetitive content
- `PunctuationFilter` - Ensure proper sentence structure
- `BoilerPlateStringFilter` - Remove template/boilerplate text

### 2. Fuzzy Deduplication

For production datasets, fuzzy deduplication is essential to remove near-duplicate content across sources:

**Key Components:**

- `FuzzyDeduplicationWorkflow` - End-to-end fuzzy deduplication pipeline
- Ray distributed computing framework for scalability
- Connected components clustering for duplicate identification

### 3. Content Cleaning

Basic text normalization and cleaning operations:

**Common Cleaning Steps:**

- `UnicodeReformatter` - Normalize Unicode characters
- `NewlineNormalizer` - Standardize line breaks
- Basic HTML/markup removal
- Note: PII removal requires specialized processing tools (see PII documentation)

### 4. Exact Deduplication

Remove identical documents, especially useful for smaller datasets:

**Implementation:**

- `ExactDuplicates` - Hash-based exact matching
- MD5 or SHA-256 hashing for document identification

## Core Processing Architecture

NeMo Curator uses these fundamental building blocks that users combine into pipelines:

```{list-table}
:header-rows: 1

* - Component
  - Purpose  
  - Usage Pattern
* - **`Pipeline`**
  - Orchestrate processing stages
  - Every workflow starts here
* - **`ScoreFilter`**
  - Apply filters with optional scoring
  - Chain multiple quality filters
* - **`Modify`**
  - Transform document content
  - Clean and normalize text
* - **Reader/Writer Stages**
  - Load and save text data
  - Input/output for pipelines
* - **Processing Stages**
  - Transform DocumentBatch tasks
  - Core processing components
```

## Key Architecture Distinctions

Understanding these core concepts helps you design effective text curation workflows:

**Tasks**
: The unit of data flowing through pipelines. In text processing, this is typically a `DocumentBatch` containing multiple documents with their metadata.

**Stages** 
: Individual processing units that perform a single operation (reading, filtering, modifying, writing). Stages transform tasks and can be chained together.

**Pipelines**
: Generic orchestration containers that execute stages in sequence. You build pipelines by adding stages: `pipeline.add_stage(reader)`, `pipeline.add_stage(filter)`.

**Workflows**
: Pre-built, specialized classes for complex multi-stage operations. Examples include `FuzzyDeduplicationWorkflow` and `ExactDeduplicationWorkflow` that encapsulate entire deduplication processes.

:::{seealso}
For more detailed information about these abstractions and their technical implementation, refer to [Key Abstractions](about-concepts-video-abstractions), which provides comprehensive coverage of NeMo Curator's core architecture patterns across all modalities.
:::

## Implementation Examples

### Complete Quality Filtering Pipeline

This is the most common starting workflow, used in 90% of production pipelines:

:::{dropdown} Quality Filtering Pipeline Code Example
:icon: code-square

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import (
    WordCountFilter,
    NonAlphaNumericFilter,
    RepeatedLinesFilter,
    PunctuationFilter,
    BoilerPlateStringFilter
)

# Create processing pipeline
pipeline = Pipeline(name="quality_filtering")

# Load dataset - the starting point for all workflows
reader = JsonlReader(file_paths="data/*.jsonl")
pipeline.add_stage(reader)

# Standard quality filtering pipeline (most common)
# Remove too short/long documents (essential)
word_count_filter = ScoreFilter(
    score_fn=WordCountFilter(min_words=50, max_words=100000),
    text_field="text",
    score_field="word_count"
)
pipeline.add_stage(word_count_filter)

# Remove symbol-heavy content
alpha_numeric_filter = ScoreFilter(
    score_fn=NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
    text_field="text"
)
pipeline.add_stage(alpha_numeric_filter)

# Remove repetitive content
repeated_lines_filter = ScoreFilter(
    score_fn=RepeatedLinesFilter(max_repeated_line_fraction=0.7),
    text_field="text"
)
pipeline.add_stage(repeated_lines_filter)

# Ensure proper sentence structure
punctuation_filter = ScoreFilter(
    score_fn=PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85),
    text_field="text"
)
pipeline.add_stage(punctuation_filter)

# Remove template/boilerplate text
boilerplate_filter = ScoreFilter(
    score_fn=BoilerPlateStringFilter(),
    text_field="text"
)
pipeline.add_stage(boilerplate_filter)

# Add writer stage
writer = JsonlWriter(path="filtered_data/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()
```

:::

### Content Cleaning Pipeline

Basic text normalization:

:::{dropdown} Content Cleaning Pipeline Code Example
:icon: code-square

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import Modify
from nemo_curator.stages.text.modifiers import UnicodeReformatter

# Create cleaning pipeline
pipeline = Pipeline(name="content_cleaning")

# Read input data
reader = JsonlReader(file_paths="input_data/*.jsonl")
pipeline.add_stage(reader)

# Essential cleaning steps
# Normalize unicode characters (very common)
unicode_modifier = Modify(
    modifier=UnicodeReformatter(),
    text_field="text"
)
pipeline.add_stage(unicode_modifier)

# Note: For PII removal, use dedicated PII processing tools
# See the PII processing documentation for specialized workflows

# Write cleaned data
writer = JsonlWriter(path="cleaned_data/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()
```

:::

### Large-Scale Fuzzy Deduplication

Critical for production datasets (requires Ray + GPU):

:::{dropdown} Fuzzy Deduplication Code Example
:icon: code-square

```python
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

# Initialize Ray cluster with GPU support (required for fuzzy deduplication)
ray_client = RayClient(num_gpus=4)
ray_client.start()

# Configure fuzzy deduplication workflow (production settings)
fuzzy_workflow = FuzzyDeduplicationWorkflow(
    input_path="/path/to/input/data",
    cache_path="./cache",
    output_path="./output",
    text_field="text",
    perform_removal=False,  # Currently only identification supported
    # LSH parameters for ~80% similarity threshold
    num_bands=20,           # Number of LSH bands
    minhashes_per_band=13,  # Hashes per band
    char_ngrams=24,         # Character n-gram size
    seed=42
)

# Run fuzzy deduplication workflow
fuzzy_workflow.run()

# Cleanup Ray when done
ray_client.stop()
```

:::

### Exact Deduplication (All dataset sizes)

Quick deduplication for any dataset size (requires Ray + GPU):

:::{dropdown} Exact Deduplication Code Example
:icon: code-square

```python
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

# Initialize Ray cluster with GPU support (required for exact deduplication)
ray_client = RayClient(num_gpus=4)
ray_client.start()

# Configure exact deduplication workflow
exact_workflow = ExactDeduplicationWorkflow(
    input_path="/path/to/input/data",
    output_path="/path/to/output",
    text_field="text",
    perform_removal=False,  # Currently only identification supported
    assign_id=True,         # Automatically assign unique IDs
    input_filetype="parquet"
)

# Run exact deduplication workflow
exact_workflow.run()

# Cleanup Ray when done
ray_client.stop()
```

:::

### Complete End-to-End Pipeline

Most users combine these steps into a comprehensive workflow:

:::{dropdown} Complete End-to-End Pipeline Code Example
:icon: code-square

```python
from nemo_curator.pipeline import Pipeline

from nemo_curator.core.client import RayClient

# Complete production pipeline (most common pattern)
def build_production_pipeline():
    pipeline = Pipeline(name="production_processing")
    
    # 1. Content cleaning first
    unicode_modifier = Modify(
        modifier=UnicodeReformatter(),
        text_field="text"
    )
    pipeline.add_stage(unicode_modifier)
    
    # Note: PII processing requires specialized tools - see PII documentation
    # for proper implementation using dedicated PII processing pipelines
    
    # 2. Quality filtering
    word_filter = ScoreFilter(
        score_fn=WordCountFilter(min_words=50, max_words=100000),
        text_field="text"
    )
    pipeline.add_stage(word_filter)
    
    alpha_filter = ScoreFilter(
        score_fn=NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
        text_field="text"
    )
    pipeline.add_stage(alpha_filter)
    
    repeated_filter = ScoreFilter(
        score_fn=RepeatedLinesFilter(max_repeated_line_fraction=0.7),
        text_field="text"
    )
    pipeline.add_stage(repeated_filter)
    
    boilerplate_filter = ScoreFilter(
        score_fn=BoilerPlateStringFilter(),
        text_field="text"
    )
    pipeline.add_stage(boilerplate_filter)
    
    return pipeline

# Apply the complete pipeline
complete_pipeline = build_production_pipeline()
processed_results = complete_pipeline.run()

# Then apply deduplication separately for large datasets
# For large datasets - use fuzzy deduplication
ray_client = RayClient(num_gpus=4)
ray_client.start()
fuzzy_workflow = FuzzyDeduplicationWorkflow(
    input_path="/path/to/processed/data",
    cache_path="./cache",
    output_path="./output",
    text_field="text"
)
fuzzy_workflow.run()

# For smaller datasets - use exact deduplication
exact_workflow = ExactDeduplicationWorkflow(
    input_path="/path/to/processed/data",
    output_path="./output",
    text_field="text",
    assign_id=True
)
exact_workflow.run()

ray_client.stop()
```

:::
