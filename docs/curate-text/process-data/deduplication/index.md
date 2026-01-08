---
description: "Remove duplicate and near-duplicate documents efficiently using GPU-accelerated and semantic deduplication modules"
categories: ["workflows"]
tags: ["deduplication", "fuzzy-dedup", "semantic-dedup", "exact-dedup", "gpu-accelerated", "minhash"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "explanation"
modality: "text-only"
---

(text-process-data-dedup)=

# Deduplication

Remove duplicate and near-duplicate documents from text datasets using NeMo Curator's GPU-accelerated deduplication workflows. Removing duplicates prevents overrepresentation of repeated content in language model training.

NeMo Curator provides three deduplication approaches: exact matching (MD5 hashing), fuzzy matching (MinHash + LSH), and semantic matching (embeddings). All methods are GPU-accelerated and integrate with the {ref}`data processing pipeline <about-concepts-text-data-processing>`.

## How It Works

NeMo Curator provides three deduplication approaches, each optimized for different duplicate types:

:::::{tab-set}

::::{tab-item} Exact

**Method**: MD5 hashing  
**Detects**: Character-for-character identical documents  
**Speed**: Fastest

Computes MD5 hashes for each document's text content and groups documents with identical hashes. Best for removing exact copies.

:::{dropdown} Code Example
:icon: code-square

```python
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

ray_client = RayClient()
ray_client.start()

exact_workflow = ExactDeduplicationWorkflow(
    input_path="/path/to/input/data",
    output_path="/path/to/output",
    text_field="text",
    perform_removal=False,  # Identification only
    assign_id=True,
    input_filetype="parquet"
)
exact_workflow.run()
```

For removal, use `TextDuplicatesRemovalWorkflow` with the generated duplicate IDs. See {ref}`Exact Duplicate Removal <text-process-data-dedup-exact>` for details.
:::

::::

::::{tab-item} Fuzzy

**Method**: MinHash + Locality Sensitive Hashing (LSH)  
**Detects**: Near-duplicates with minor edits (~80% similarity)  
**Speed**: Fast

Generates MinHash signatures and uses LSH to find similar documents. Best for detecting documents with small formatting differences or typos.

:::{dropdown} Code Example
:icon: code-square

```python
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

ray_client = RayClient()
ray_client.start()

fuzzy_workflow = FuzzyDeduplicationWorkflow(
    input_path="/path/to/input/data",
    cache_path="/path/to/cache",
    output_path="/path/to/output",
    text_field="text",
    perform_removal=False,  # Identification only
    input_blocksize="1GiB",
    seed=42,
    char_ngrams=24,
    num_bands=20,
    minhashes_per_band=13
)
fuzzy_workflow.run()
```

For removal, use `TextDuplicatesRemovalWorkflow` with the generated duplicate IDs. See {ref}`Fuzzy Duplicate Removal <text-process-data-dedup-fuzzy>` for details.
:::

::::

::::{tab-item} Semantic

**Method**: Embeddings + clustering + pairwise similarity  
**Detects**: Semantically similar content (paraphrases, translations)  
**Speed**: Moderate

Generates embeddings using transformer models, clusters them, and computes pairwise similarities. Best for meaning-based deduplication.

:::{dropdown} Code Example
:icon: code-square

```python
from nemo_curator.stages.text.deduplication.semantic import TextSemanticDeduplicationWorkflow

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
text_workflow.run()
```

**Note**: Two workflows available:

- `TextSemanticDeduplicationWorkflow`: For raw text with automatic embedding generation
- `SemanticDeduplicationWorkflow`: For pre-computed embeddings

See {ref}`Semantic Deduplication <text-process-data-dedup-semdedup>` for details.
:::

:::{dropdown} Advanced: Step-by-Step Semantic Deduplication
:icon: gear

For fine-grained control, break semantic deduplication into separate stages:

```python
from nemo_curator.stages.deduplication.id_generator import create_id_generator_actor
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow

# 1. Create ID generator
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
# 5. Identify and remove duplicates
```

This approach enables analysis of intermediate results and fine-grained control.
:::

::::

:::::

---

## Deduplication Methods

Choose a deduplication method based on your needs:

::::{grid} 1 1 1 2
:gutter: 2

:::{grid-item-card} {octicon}`git-pull-request;1.5em;sd-mr-1` Exact Duplicate Removal
:link: exact
:link-type: doc
Identify and remove character-for-character duplicates using MD5 hashing
+++
{bdg-secondary}`hashing`
{bdg-secondary}`fast`
{bdg-secondary}`gpu-accelerated`
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` Fuzzy Duplicate Removal
:link: fuzzy
:link-type: doc
Identify and remove near-duplicates using MinHash and LSH similarity
+++
{bdg-secondary}`minhash`
{bdg-secondary}`lsh`
{bdg-secondary}`gpu-accelerated`
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

Duplicate removal workflows require stable document identifiers. Choose one approach:

- **Use `AddId`** to add IDs at the start of your pipeline
- **Use reader-based ID generation** (`_generate_ids`, `_assign_ids`) backed by the ID Generator actor for stable integer IDs
- **Use existing IDs** if your documents already have unique identifiers

Some workflows write an ID generator state file (`*_id_generator.json`) for later removal when IDs are auto-assigned.

### Removing Duplicates

Use `TextDuplicatesRemovalWorkflow` to apply duplicate IDs to your original dataset. Works with IDs from exact, fuzzy, or semantic deduplication.

```python
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow

removal_workflow = TextDuplicatesRemovalWorkflow(
    input_path="/path/to/input",
    ids_to_remove_path="/path/to/duplicates",  # ExactDuplicateIds/, FuzzyDuplicateIds/, or duplicates/
    output_path="/path/to/clean",
    input_filetype="parquet",
    input_id_field="_curator_dedup_id",
    ids_to_remove_duplicate_id_field="_curator_dedup_id",
    id_generator_path="/path/to/id_generator.json"  # Required when IDs were auto-assigned
)
removal_workflow.run()
```

:::{dropdown} ID Field Configuration
:icon: gear

**When `assign_id=True`** (IDs auto-assigned):

- Duplicate IDs file contains `_curator_dedup_id` column
- Set `ids_to_remove_duplicate_id_field="_curator_dedup_id"`
- `id_generator_path` is required

**When `assign_id=False`** (using existing IDs):

- Duplicate IDs file contains the column specified by `id_field` (e.g., `"id"`)
- Set `ids_to_remove_duplicate_id_field` to match your `id_field` value
- `id_generator_path` not required
:::

### Outputs and Artifacts

Each deduplication method produces specific output files and directories:

```{list-table} Output Locations
:header-rows: 1
:name: output-locations

* - Method
  - Duplicate IDs Location
  - ID Generator File
  - Deduplicated Output
* - Exact
  - `ExactDuplicateIds/` (parquet)
  - `exact_id_generator.json` (if `assign_id=True`)
  - Via `TextDuplicatesRemovalWorkflow`
* - Fuzzy
  - `FuzzyDuplicateIds/` (parquet)
  - `fuzzy_id_generator.json` (if IDs auto-assigned)
  - Via `TextDuplicatesRemovalWorkflow`
* - Semantic
  - `output_path/duplicates/` (parquet)
  - N/A
  - `output_path/deduplicated/` (if `perform_removal=True`)
```

**Column names**:

- `_curator_dedup_id` when `assign_id=True` or IDs are auto-assigned
- Matches `id_field` parameter when `assign_id=False`

## Choosing a Deduplication Method

Compare deduplication methods to select the best approach for your dataset:

```{list-table} Method Comparison
:header-rows: 1
:name: method-comparison

* - Method
  - Best For
  - Speed
  - Duplicate Types
  - GPU Required
* - **Exact**
  - Identical copies
  - Very fast
  - Character-for-character matches
  - Required
* - **Fuzzy**
  - Near-duplicates with small changes
  - Fast
  - Minor edits, reformatting (~80% similarity)
  - Required
* - **Semantic**
  - Similar meaning, different words
  - Moderate
  - Paraphrases, translations, rewrites
  - Required
```

### Quick Decision Guide

Use this guide to quickly select the right method:

- **Start with Exact** if you have numerous identical documents or need the fastest speed
- **Use Fuzzy** if you need to catch near-duplicates with minor formatting differences
- **Use Semantic** for meaning-based deduplication on large, diverse datasets

:::{dropdown} When to Use Each Method
:icon: info

**Exact Deduplication**:

- Removing identical copies of documents
- Fast initial deduplication pass
- Datasets with numerous exact duplicates
- When speed is more important than detecting near-duplicates

**Fuzzy Deduplication**:

- Removing near-duplicate documents with minor formatting differences
- Detecting documents with small edits or typos
- Fast deduplication when exact matching misses numerous duplicates
- When speed is important but some near-duplicate detection is needed

**Semantic Deduplication**:

- Removing semantically similar content (paraphrases, translations)
- Large, diverse web-scale datasets
- When meaning-based deduplication is more important than speed
- Advanced use cases requiring embedding-based similarity detection
:::

:::{dropdown} Combining Methods
:icon: git-branch

You can combine deduplication methods for comprehensive duplicate removal:

1. **Exact → Fuzzy → Semantic**: Start with fastest methods, then apply more sophisticated methods
2. **Exact → Semantic**: Use exact for quick wins, then semantic for meaning-based duplicates
3. **Fuzzy → Semantic**: Use fuzzy for near-duplicates, then semantic for paraphrases

Run each method independently, then combine duplicate IDs before removal.
:::

For detailed implementation guides, see:

- {ref}`Exact Duplicate Removal <text-process-data-dedup-exact>`
- {ref}`Fuzzy Duplicate Removal <text-process-data-dedup-fuzzy>`
- {ref}`Semantic Deduplication <text-process-data-dedup-semdedup>`

:::{dropdown} Performance Considerations
:icon: zap

### GPU Acceleration

All deduplication workflows require GPU acceleration:

- **Exact**: Ray backend with GPU support for MD5 hashing operations
- **Fuzzy**: Ray backend with GPU support for MinHash computation and LSH operations
- **Semantic**: GPU required for embedding generation (transformer models), K-means clustering, and pairwise similarity computation

GPU acceleration provides significant speedup for large datasets through parallel processing.

### Hardware Requirements

- **GPU**: Required for all workflows (Ray with GPU support for exact/fuzzy, GPU for semantic)
- **Memory**: GPU memory requirements scale with dataset size, batch sizes, and embedding dimensions
- **Executors**: Can use various executors (XennaExecutor, RayDataExecutor) with GPU support

### Backend Setup

For optimal performance with large datasets, configure Ray backend:

```python
from nemo_curator.core.client import RayClient

client = RayClient(
    num_cpus=64,    # Adjust based on available cores
    num_gpus=4      # Should be roughly 2x the memory of embeddings
)
client.start()

try:
    workflow.run()
finally:
    client.stop()
```

For TB-scale datasets, consider distributed GPU clusters with Ray.

### ID Generator for Large-Scale Operations

For large-scale duplicate removal, persist the ID Generator for consistent document tracking:

```python
from nemo_curator.stages.deduplication.id_generator import (
    create_id_generator_actor,
    write_id_generator_to_disk,
    kill_id_generator_actor
)

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

The ID Generator ensures consistent IDs across workflow stages.
:::

## Next Steps

**Ready to use deduplication?**

- **New to deduplication**: Start with {ref}`Exact Duplicate Removal <text-process-data-dedup-exact>` for the fastest approach
- **Need near-duplicate detection**: See {ref}`Fuzzy Duplicate Removal <text-process-data-dedup-fuzzy>` for MinHash-based matching
- **Require semantic matching**: Explore {ref}`Semantic Deduplication <text-process-data-dedup-semdedup>` for meaning-based deduplication

**For hands-on guidance**: See {ref}`Text Curation Tutorials <text-tutorials>` for step-by-step examples.

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Exact Duplicate Removal <exact>
Fuzzy Duplicate Removal <fuzzy>
Semantic Deduplication <semdedup>
```
