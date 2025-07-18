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
  - Optional
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
from nemo_curator import ExactDuplicates, FuzzyDuplicates, SemDedup
from nemo_curator.datasets import DocumentDataset

# Load your dataset
# Note: Use "cudf" backend for GPU acceleration, "pandas" for CPU
dataset = DocumentDataset.read_json("input_data/*.jsonl", backend="cudf")

# Option 1: Exact deduplication (CPU/GPU flexible)
exact_dedup = ExactDuplicates(
    id_field="doc_id",
    text_field="text",
    perform_removal=True
)
# Works with both "cudf" (GPU) and "pandas" (CPU) backends
deduplicated = exact_dedup(dataset)

# Option 2: Fuzzy deduplication (requires GPU)
from nemo_curator import FuzzyDuplicatesConfig
fuzzy_config = FuzzyDuplicatesConfig(
    cache_dir="./fuzzy_cache",
    id_field="doc_id", 
    text_field="text",
    perform_removal=True
)
fuzzy_dedup = FuzzyDuplicates(config=fuzzy_config)
# Requires cudf backend (GPU)
deduplicated = fuzzy_dedup(dataset)

# Option 3: Semantic deduplication (requires GPU)
from nemo_curator import SemDedupConfig
sem_config = SemDedupConfig(
    cache_dir="./sem_cache",
    embedding_model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"
)
sem_dedup = SemDedup(config=sem_config, id_column="doc_id", perform_removal=True)
# Requires cudf backend (GPU)
deduplicated = sem_dedup(dataset)
```

## Performance Considerations

### GPU Acceleration

- **Exact deduplication**: Supports both CPU and GPU backends. GPU provides significant speedup for large datasets through optimized hashing operations
- **Fuzzy deduplication**: Requires GPU backend for MinHash and LSH operations. GPU acceleration is essential for processing large datasets efficiently
- **Semantic deduplication**: Requires GPU backend for embedding generation and clustering operations. GPU acceleration is critical for feasible processing times

### Hardware Requirements

- **CPU-only workflows**: Only exact deduplication is available
- **GPU workflows**: All three methods available. Recommended for large-scale data processing
- **Memory considerations**: GPU memory requirements scale with dataset size and embedding dimensions

For very large datasets (TB-scale), consider running deduplication on distributed GPU clusters.

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Hash-Based Deduplication <gpudedup>
Semantic Deduplication <semdedup>
``` 