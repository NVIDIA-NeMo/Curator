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

Removing duplicates improves language model training by preventing overrepresentation of repeated content. NeMo Curator provides multiple approaches to deduplication, from exact hash-based matching to semantic similarity detection using embeddings.

## How It Works

NeMo Curator offers three main approaches to deduplication:

1. **Exact Deduplication**: Uses document hashing to identify identical content
2. **Fuzzy Deduplication**: Uses MinHash and LSH to find near-duplicate content  
3. **Semantic Deduplication**: Uses embeddings to identify semantically similar content

Each approach serves different use cases and offers different trade-offs between speed, accuracy, and the types of duplicates detected.

---

## Deduplication Methods

::::{grid} 1 1 1 2
:gutter: 2

:::{grid-item-card} {octicon}`git-pull-request;1.5em;sd-mr-1` Hash-Based Deduplication
:link: gpudedup
:link-type: doc
Remove exact and fuzzy duplicates using hashing algorithms
+++
{bdg-secondary}`minhash`
{bdg-secondary}`lsh`
{bdg-secondary}`hashing`
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
    # MinHash + LSH parameters
    seed=42,
    char_ngrams=24,
    num_bands=20,
    minhashes_per_band=13
)
fuzzy_workflow.run()

# Option 3: Semantic deduplication (requires GPU)
# For text with embedding generation
from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow

text_sem_workflow = SemanticDeduplicationWorkflow(
    input_path="/path/to/input/data",
    output_path="/path/to/output", 
    cache_path="/path/to/cache",
    text_field="text",
    model_identifier="sentence-transformers/all-MiniLM-L6-v2",
    n_clusters=100,
    perform_removal=False  # Currently only identification supported
)
# Requires executor (Xenna or RayData)
from nemo_curator.backends.xenna import XennaExecutor
text_sem_workflow.run(XennaExecutor())

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
sem_workflow.run(pairwise_executor=XennaExecutor())
```

## Performance Considerations

### GPU Acceleration

- **Exact deduplication**: Requires Ray backend with GPU support for MD5 hashing operations. GPU acceleration provides significant speedup for large datasets
- **Fuzzy deduplication**: Requires Ray backend with GPU support for MinHash and LSH operations. GPU acceleration is essential for processing large datasets efficiently
- **Semantic deduplication**: Requires GPU backend for embedding generation and clustering operations. GPU acceleration is critical for feasible processing times

### Hardware Requirements

- **CPU-only workflows**: No deduplication workflows available (all require Ray + GPU)
- **Ray + GPU workflows**: All deduplication methods require Ray distributed computing framework with GPU support
- **Memory considerations**: GPU memory requirements scale with dataset size and embedding dimensions

For very large datasets (TB-scale), consider running deduplication on distributed GPU clusters with Ray.

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Hash-Based Deduplication <gpudedup>
Semantic Deduplication <semdedup>
```