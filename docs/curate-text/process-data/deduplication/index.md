---
description: "Remove duplicate and near-duplicate documents efficiently using GPU-accelerated and semantic deduplication modules"
categories: ["workflows"]
tags: ["deduplication", "fuzzy-dedup", "semantic-dedup", "exact-dedup", "gpu-accelerated", "minhash"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(text-process-data-dedup)=

# Deduplication

Remove duplicate and near-duplicate documents efficiently from your text datasets using NeMo Curator's GPU-accelerated and semantic deduplication modules.

Removing duplicates improves language model training by preventing overrepresentation of repeated content. NeMo Curator provides multiple approaches to deduplication, from exact hash-based matching to semantic similarity detection using embeddings. These workflows are part of the comprehensive {ref}`data processing pipeline <about-concepts-text-data-processing>`.

## How It Works

NeMo Curator's deduplication framework is built around three main approaches that work within the {ref}`data processing architecture <about-concepts-text-data-processing>`:

::::{tab-set}

:::{tab-item} Exact

Exact deduplication uses MD5 hashing to identify identical documents:

```python
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

# Configure exact deduplication
exact_workflow = ExactDeduplicationWorkflow(
    input_path="/path/to/input/data",
    output_path="/path/to/output",
    text_field="text",
    perform_removal=False,  # Currently only identification supported
    assign_id=True,  # Automatically assign unique IDs
    input_filetype="parquet"  # "parquet" or "jsonl"
)

# Run with Ray backend (GPU required)
exact_workflow.run()
```

The workflow:

1. Computes MD5 hashes for each document's text content
2. Groups documents by identical hash values
3. Identifies duplicates for removal or creates cleaned dataset

:::

:::{tab-item} Fuzzy

Fuzzy deduplication uses MinHash and LSH to find near-duplicate content:

```python
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

# Configure fuzzy deduplication
fuzzy_workflow = FuzzyDeduplicationWorkflow(
    input_path="/path/to/input/data",
    cache_path="/path/to/cache",
    output_path="/path/to/output",
    text_field="text",
    perform_removal=False,  # Currently only identification supported
    input_blocksize="1GiB",  # Default block size (differs from exact dedup)
    # MinHash + LSH parameters
    seed=42,
    char_ngrams=24,
    num_bands=20,
    minhashes_per_band=13
)

# Run with Ray backend (GPU required)
fuzzy_workflow.run()
```

The workflow:

1. Generates MinHash signatures for each document
2. Uses Locality Sensitive Hashing (LSH) to find similar signatures
3. Identifies near-duplicates based on similarity thresholds

:::

:::{tab-item} Semantic

Semantic deduplication uses embeddings to identify meaning-based duplicates:

```python
from nemo_curator.stages.text.deduplication.semantic import TextSemanticDeduplicationWorkflow

# End-to-end semantic deduplication
text_workflow = TextSemanticDeduplicationWorkflow(
    input_path="/path/to/input/data",
    output_path="/path/to/output", 
    cache_path="/path/to/cache",
    text_field="text",
    model_identifier="sentence-transformers/all-MiniLM-L6-v2",
    n_clusters=100,
    eps=0.01,  # Similarity threshold
    perform_removal=True  # Complete deduplication
)

# Run with GPU backend
text_workflow.run()
```

The workflow:

1. Generates embeddings for each document using transformer models
2. Clusters embeddings using K-means
3. Computes pairwise similarities within clusters
4. Identifies semantic duplicates based on cosine similarity threshold

**Note**: Semantic deduplication offers two workflows:

- `TextSemanticDeduplicationWorkflow`: For raw text input with automatic embedding generation
- `SemanticDeduplicationWorkflow`: For pre-computed embeddings

:::

:::{tab-item} Step-by-Step

For advanced users, semantic deduplication can be broken down into separate stages:

```python
from nemo_curator.stages.deduplication.id_generator import create_id_generator_actor
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow

# 1. Create ID generator for consistent tracking
create_id_generator_actor()

# 2. Generate embeddings separately
embedding_pipeline = Pipeline(
    stages=[
        ParquetReader(file_paths=input_path, _generate_ids=True),
        EmbeddingCreatorStage(
            model_identifier="sentence-transformers/all-MiniLM-L6-v2",
            text_field="text"
        ),
        ParquetWriter(path=embedding_output_path, fields=["_curator_dedup_id", "embeddings"])
    ]
)
embedding_out = embedding_pipeline.run()

# 3. Run clustering and pairwise similarity
semantic_workflow = SemanticDeduplicationWorkflow(
    input_path=embedding_output_path,
    output_path=semantic_workflow_path,
    n_clusters=100,
    id_field="_curator_dedup_id",
    embedding_field="embeddings",
    eps=None  # Skip duplicate identification for analysis
)
semantic_out = semantic_workflow.run()

# 4. Analyze results and choose eps parameter
# (analyze cosine similarity distributions)

# 5. Identify and remove duplicates
# (run duplicate identification and removal workflows)
```

This approach provides fine-grained control over each stage and enables analysis of intermediate results.

:::

::::

Each approach serves different use cases and offers different trade-offs between speed, accuracy, and the types of duplicates detected.

---

## Deduplication Methods

::::{grid} 1 1 1 2
:gutter: 2

:::{grid-item-card} {octicon}`git-pull-request;1.5em;sd-mr-1` Exact Duplicate Removal
:link: exact
:link-type: doc
Identify character-for-character duplicates using hashing
+++
{bdg-secondary}`hashing`
{bdg-secondary}`fast`
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` Fuzzy Duplicate Removal
:link: fuzzy
:link-type: doc
Identify near-duplicates using MinHash and LSH
+++
{bdg-secondary}`minhash`
{bdg-secondary}`lsh`
{bdg-secondary}`fast`
:::

:::{grid-item-card} {octicon}`repo-clone;1.5em;sd-mr-1` Semantic Deduplication
:link: semdedup
:link-type: doc
Remove semantically similar documents using embeddings
+++
{bdg-secondary}`embeddings`
{bdg-secondary}`gpu-accelerated`
{bdg-secondary}`meaning-based`
{bdg-secondary}`advanced`
:::

::::

(text-process-data-dedup-common-ops)=

## Common Operations

### Document IDs

Duplicate removal workflows require stable document identifiers.

- Use `AddId` to add IDs at the start of your pipeline
- Or use reader-based ID generation (`_generate_ids`, `_assign_ids`) backed by the ID Generator actor for stable integer IDs
- Some workflows write an ID generator state file for later removal

### Outputs and Artifacts

- Exact duplicate identification:
  - `ExactDuplicateIds/` (parquet with column `id`)
  - `exact_id_generator.json`
- Fuzzy duplicate identification:
  - `FuzzyDuplicateIds/` (parquet with column `id`)
  - `fuzzy_id_generator.json`
- Semantic duplicate identification/removal:
  - `output_path/duplicates/` (parquet with column `id`)
  - `output_path/deduplicated/` (when `perform_removal=True`)

### Removing Duplicates

Use the Text Duplicates Removal workflow to apply a list of duplicate IDs to your original dataset.

```python
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow

removal_workflow = TextDuplicatesRemovalWorkflow(
    input_path="/path/to/input",
    ids_to_remove_path="/path/to/duplicates",
    output_path="/path/to/clean",
    input_filetype="parquet",
    input_id_field="_curator_dedup_id",
    ids_to_remove_duplicate_id_field="id",
)

removal_workflow.run()
```

## Usage

Here's a quick comparison of the different deduplication approaches:

```{list-table} Deduplication Method Comparison
:header-rows: 1
:widths: 20 20 25 25 10

* - Method
  - Best For
  - Speed
  - Duplicate Types Detected
  - GPU Required
* - Exact Deduplication
  - Identical copies
  - Very Fast
  - Character-for-character matches
  - Required
* - Fuzzy Deduplication
  - Near-duplicates with small changes
  - Fast
  - Content with minor edits, reformatting
  - Required
* - Semantic Deduplication
  - Similar meaning, different words
  - Moderate
  - Paraphrases, translations, rewrites
  - Required
```

### Quick Start Example

```python
# Import workflows directly from their modules (not from __init__.py)
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow

# Option 1: Exact deduplication (requires Ray + GPU)
exact_workflow = ExactDeduplicationWorkflow(
    input_path="/path/to/input/data",
    output_path="/path/to/output",
    text_field="text",
    perform_removal=False,  # Currently only identification supported
    assign_id=True,  # Automatically assign unique IDs
    input_filetype="parquet"  # "parquet" or "jsonl"
)
exact_workflow.run()

# Option 2: Fuzzy deduplication (requires Ray + GPU)
fuzzy_workflow = FuzzyDeduplicationWorkflow(
    input_path="/path/to/input/data",
    cache_path="/path/to/cache",
    output_path="/path/to/output",
    text_field="text",
    perform_removal=False,  # Currently only identification supported
    input_blocksize="1GiB",  # Default block size (differs from exact dedup)
    # MinHash + LSH parameters
    seed=42,
    char_ngrams=24,
    num_bands=20,
    minhashes_per_band=13
)
fuzzy_workflow.run()

# Option 3: Semantic deduplication (requires GPU)
# For text with embedding generation
from nemo_curator.stages.text.deduplication.semantic import TextSemanticDeduplicationWorkflow

text_sem_workflow = TextSemanticDeduplicationWorkflow(
    input_path="/path/to/input/data",
    output_path="/path/to/output", 
    cache_path="/path/to/cache",
    text_field="text",
    model_identifier="sentence-transformers/all-MiniLM-L6-v2",
    n_clusters=100,
    perform_removal=False  # Set to True to remove duplicates, False to only identify
)
# Uses XennaExecutor by default for all stages
text_sem_workflow.run()

# Alternative: For pre-computed embeddings
from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow

sem_workflow = SemanticDeduplicationWorkflow(
    input_path="/path/to/embeddings/data",
    output_path="/path/to/output",
    n_clusters=100,
    id_field="id",
    embedding_field="embeddings"
)
# Requires executor for pairwise stage
sem_workflow.run()  # Uses XennaExecutor by default
```

## Performance Considerations

### GPU Acceleration

- **Exact deduplication**: Requires Ray backend with GPU support for MD5 hashing operations. GPU acceleration provides significant speedup for large datasets through parallel processing
- **Fuzzy deduplication**: Requires Ray backend with GPU support for MinHash computation and LSH operations. GPU acceleration is essential for processing large datasets efficiently
- **Semantic deduplication**:
  - `TextSemanticDeduplicationWorkflow`: Requires GPU for embedding generation (transformer models), K-means clustering, and pairwise similarity computation
  - `SemanticDeduplicationWorkflow`: Requires GPU for K-means clustering and pairwise similarity operations when working with pre-computed embeddings
  - GPU acceleration is critical for feasible processing times, especially for embedding generation and similarity computations

### Hardware Requirements

- **GPU Requirements**: All deduplication workflows require GPU acceleration for optimal performance
  - Exact and fuzzy deduplication require Ray distributed computing framework with GPU support for hash computations
  - Semantic deduplication requires GPU for transformer model inference, clustering algorithms, and similarity computations
  - Can use various executors (XennaExecutor, RayDataExecutor) with GPU support
- **Memory considerations**: GPU memory requirements scale with dataset size, batch sizes, and embedding dimensions (for semantic deduplication)

### Backend Setup

For optimal performance, especially with large datasets, configure Ray backend appropriately:

```python
from nemo_curator.core.client import RayClient

# Configure Ray cluster for deduplication workloads
client = RayClient(
    num_cpus=64,    # Adjust based on available cores
    num_gpus=4      # Should be roughly 2x the memory of embeddings
)
client.start()

try:
    # Run your deduplication workflow
    workflow.run()
finally:
    client.stop()
```

For very large datasets (TB-scale), consider running deduplication on distributed GPU clusters with Ray.

### ID Generator for Large-Scale Operations

For large-scale duplicate removal, use the ID Generator to ensure consistent document tracking:

```python
from nemo_curator.stages.deduplication.id_generator import (
    create_id_generator_actor, 
    write_id_generator_to_disk,
    kill_id_generator_actor
)

# Create and persist ID generator
create_id_generator_actor()
id_generator_path = "semantic_id_generator.json"
write_id_generator_to_disk(id_generator_path)
kill_id_generator_actor()

# Use saved ID generator in removal workflow
removal_workflow = TextDuplicatesRemovalWorkflow(
    input_path=input_path,
    ids_to_remove_path=duplicates_path,
    output_path=output_path,
    id_generator_path=id_generator_path,
    # ... other parameters
)
```

The ID Generator ensures that the same documents receive identical IDs across different workflow stages, enabling efficient duplicate removal.

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Exact Duplicate Removal <exact>
Fuzzy Duplicate Removal <fuzzy>
Semantic Deduplication <semdedup>
```
