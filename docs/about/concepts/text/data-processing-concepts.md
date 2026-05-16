---
description: "Text processing workflows including quality filtering, fuzzy deduplication, content cleaning, and pipeline design"
categories: ["concepts-architecture"]
tags: ["data-processing", "quality-filtering", "deduplication", "pipeline", "distributed"]
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
- `RepeatedLinesFilter` - Remove if content is too repetitive
- `PunctuationFilter` - Ensure proper sentence structure
- `BoilerPlateStringFilter` - Remove if content contains too much template/boilerplate text

### 2. Content Cleaning and Modification

Basic text normalization and cleaning operations:

**Common Cleaning Steps:**

- `UnicodeReformatter` - Normalize Unicode characters
- `NewlineNormalizer` - Standardize line breaks
- Basic HTML/markup removal

### 3. Deduplication

Remove duplicate and near-duplicate content. For comprehensive coverage of all deduplication approaches, refer to Curator's [Deduplication Concepts](about-concepts-deduplication).

#### Exact Deduplication

Remove identical documents, especially useful for smaller datasets:

**Implementation:** MD5 or SHA-256 hashing for document identification

#### Fuzzy Deduplication

For production datasets, fuzzy deduplication is essential to remove near-duplicate content across sources:

**Key Components:**

- Ray distributed computing framework for scalability
- Connected components clustering for duplicate identification

#### Semantic Deduplication

Remove semantically similar content using embeddings for more sophisticated duplicate detection.

## Core Processing Architecture

NeMo Curator uses these fundamental building blocks that users combine into pipelines:

```{list-table}
:header-rows: 1

* - Component
  - Purpose
  - Usage Pattern
* - **`Pipeline`**
  - Orchestrate processing stages
  - Add processing stages, typically starting with a read and completing with a write
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

## Implementation Examples

### Complete Quality Filtering Pipeline

This is the most common starting workflow, used in 90% of production pipelines:

:::{dropdown} Quality Filtering Pipeline Code Example
:icon: code-square

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.filters.heuristic.repetition import RepeatedLinesFilter
from nemo_curator.stages.text.filters.heuristic import (
    WordCountFilter,
    NonAlphaNumericFilter,
    PunctuationFilter,
    BoilerPlateStringFilter
)

# Start Ray client
ray_client = RayClient()
ray_client.start()

# Create processing pipeline
pipeline = Pipeline(name="quality_filtering")

# Load dataset - the starting point for all workflows
reader = JsonlReader(file_paths="input_data/")
pipeline.add_stage(reader)

# Standard quality filtering pipeline (most common)
# Remove too short/long documents (essential)
# and save the word_count field
word_count_filter = ScoreFilter(
    filter_obj=WordCountFilter(min_words=50, max_words=100000),
    text_field="text",
    score_field="word_count"
)
pipeline.add_stage(word_count_filter)

# Remove symbol-heavy content
alpha_numeric_filter = ScoreFilter(
    filter_obj=NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
    text_field="text"
)
pipeline.add_stage(alpha_numeric_filter)

# Remove repetitive content
repeated_lines_filter = ScoreFilter(
    filter_obj=RepeatedLinesFilter(max_repeated_line_fraction=0.7),
    text_field="text"
)
pipeline.add_stage(repeated_lines_filter)

# Ensure proper sentence structure
punctuation_filter = ScoreFilter(
    filter_obj=PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85),
    text_field="text"
)
pipeline.add_stage(punctuation_filter)

# Remove template/boilerplate text
boilerplate_filter = ScoreFilter(
    filter_obj=BoilerPlateStringFilter(),
    text_field="text"
)
pipeline.add_stage(boilerplate_filter)

# Add writer stage
writer = JsonlWriter(path="filtered_data/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()

# Cleanup Ray when done
ray_client.stop()
```

:::

### Content Cleaning Pipeline

Basic text normalization:

:::{dropdown} Content Cleaning Pipeline Code Example
:icon: code-square

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import Modify
from nemo_curator.stages.text.modifiers.unicode import UnicodeReformatter

# Start Ray client
ray_client = RayClient()
ray_client.start()

# Create cleaning pipeline
pipeline = Pipeline(name="content_cleaning")

# Read input data
reader = JsonlReader(file_paths="input_data/")
pipeline.add_stage(reader)

# Essential cleaning steps
# Normalize unicode characters (very common)
unicode_modifier = Modify(
    modifier_fn=UnicodeReformatter(),
    input_fields="text"
)
pipeline.add_stage(unicode_modifier)

# Additional processing steps can be added as needed

# Write cleaned data
writer = JsonlWriter(path="cleaned_data/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()

# Cleanup Ray when done
ray_client.stop()
```

:::

### Exact Deduplication Workflow

Exact deduplication for any dataset size (requires Ray and at least 1 GPU):

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
    input_filetype="parquet",
)

# Run exact deduplication workflow
exact_workflow.run()

# Cleanup Ray when done
ray_client.stop()
```

:::

### Fuzzy Deduplication Workflow

Critical for production datasets (requires Ray and at least 1 GPU):

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
    cache_path="/path/to/cache",
    output_path="/path/to/output",
    input_filetype="parquet",
    input_blocksize="1.5GiB",
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

### Removing Identified Duplicates

The identified duplicates can be removed using a separate workflow:

:::{dropdown} Duplicate Removal Code Example
:icon: code-square

```python
from nemo_curator.core.client import RayClient
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow

# Start Ray client
ray_client = RayClient()
ray_client.start()

# Configure workflow with input dataset and output duplicate IDs
removal_workflow = TextDuplicatesRemovalWorkflow(
    input_path="/path/to/input/data",
    ids_to_remove_path="/path/to/output/FuzzyDuplicateIds",
    output_path="/path/to/deduplicated/output",
    input_filetype="parquet",  # Same as identification workflow
    input_blocksize="1.5GiB",  # Same as identification workflow
    duplicate_id_field="_curator_dedup_id",
    id_generator_path="/path/to/output/fuzzy_id_generator.json",
)

# Run removal workflow
removal_workflow.run()

# Cleanup Ray when done
ray_client.stop()
```

:::
