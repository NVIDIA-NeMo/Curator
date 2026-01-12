---
description: "Remove semantically redundant data using embeddings and clustering to identify meaning-based duplicates in large text datasets"
categories: ["how-to-guides"]
tags: ["semantic-dedup", "embeddings", "clustering", "similarity", "meaning-based", "advanced"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-format-sem-dedup)=
# Semantic Deduplication

Detect and remove semantically redundant data from your large text datasets using NeMo Curator.

Unlike exact or fuzzy deduplication, which focus on textual similarity, semantic deduplication leverages the meaning of content to identify duplicates. This approach can significantly reduce dataset size while maintaining or even improving model performance.

The technique uses embeddings to identify "semantic duplicates" - content pairs that convey similar meaning despite using different words.

:::{note}
**GPU Acceleration**: Semantic deduplication requires GPU acceleration for both embedding generation and clustering operations. This method uses cuDF for GPU-accelerated dataframe operations and PyTorch models on GPU for optimal performance.
:::

## How It Works

Semantic deduplication identifies meaning-based duplicates using embeddings:

1. Generates embeddings for each document using transformer models
2. Clusters embeddings using K-means
3. Computes pairwise cosine similarities within clusters
4. Identifies semantic duplicates based on similarity threshold
5. Removes duplicates, keeping one representative per group

:::{note}
Based on [SemDeDup: Data-efficient learning at web-scale through semantic deduplication](https://arxiv.org/pdf/2303.09540) by Abbas et al.
:::

## Before You Start

**Prerequisites**:

- GPU acceleration (required for embedding generation and clustering)
- Stable document identifiers for removal (either existing IDs or IDs managed by the workflow and removal stages)

## Quick Start

Get started with semantic deduplication using the following example of identifying duplicates, then remove them in one step:

```python
from nemo_curator.stages.text.deduplication.semantic import TextSemanticDeduplicationWorkflow
from nemo_curator.backends.experimental.ray_data import RayDataExecutor

workflow = TextSemanticDeduplicationWorkflow(
    input_path="input_data/",
    output_path="./results", 
    cache_path="./sem_cache",
    model_identifier="sentence-transformers/all-MiniLM-L6-v2",
    n_clusters=100,
    eps=0.07,  # Similarity threshold
    id_field="doc_id",
    perform_removal=True  # Complete deduplication
)

executor = RayDataExecutor()
results = workflow.run(executor)
# Clean dataset saved to ./results/deduplicated/
```

## Configuration

Configure semantic deduplication using these key parameters:

:::{dropdown} Step-by-Step Workflow
:icon: code-square

For fine-grained control, break semantic deduplication into separate stages:

```python
from nemo_curator.stages.deduplication.id_generator import create_id_generator_actor
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow, IdentifyDuplicatesStage
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

# Step 1: Create ID generator
create_id_generator_actor()

# Step 2: Generate embeddings separately
embedding_pipeline = Pipeline(
    name="embedding_pipeline",
    stages=[
        ParquetReader(file_paths=input_path, files_per_partition=1, fields=["text"], _generate_ids=True),
        EmbeddingCreatorStage(
            model_identifier="sentence-transformers/all-MiniLM-L6-v2",
            text_field="text",
            embedding_pooling="mean_pooling",
            model_inference_batch_size=256,
        ),
        ParquetWriter(path=embedding_output_path, fields=["_curator_dedup_id", "embeddings"]),
    ],
)
embedding_out = embedding_pipeline.run()

# Step 3: Run clustering and pairwise similarity (without duplicate identification)
semantic_workflow = SemanticDeduplicationWorkflow(
    input_path=embedding_output_path,
    output_path=semantic_workflow_path,
    n_clusters=100,
    id_field="_curator_dedup_id",
    embedding_field="embeddings",
    eps=None,  # Skip duplicate identification for analysis
)
semantic_out = semantic_workflow.run()

# Step 4: Analyze similarity distribution to choose eps
# Step 5: Identify duplicates with chosen eps
# Step 6: Remove duplicates from original dataset
```

This approach enables analysis of intermediate results and parameter tuning.
:::

### Comparison with Other Deduplication Methods

Compare semantic deduplication with other methods:

```{list-table} Deduplication Method Behavior Comparison
:header-rows: 1
:widths: 25 25 25 25

* - Method
  - Return Value Options
  - perform_removal Parameter
  - Workflow
* - ExactDuplicates
  - Duplicates (ID list only)
  - ❌ Not supported (must remain `False`; use `TextDuplicatesRemovalWorkflow`)
  - Two-step (identification + removal workflow)
* - FuzzyDuplicates
  - Duplicates (ID list only)  
  - ❌ Not supported (must remain `False`; use `TextDuplicatesRemovalWorkflow`)
  - Two-step (identification + removal workflow)
* - TextSemanticDeduplicationWorkflow
  - Duplicates or Clean Dataset
  - ✅ Available
  - One-step or two-step
```

### Key Parameters

```{list-table} Key Configuration Parameters
:header-rows: 1
:widths: 25 15 20 40

* - Parameter
  - Type
  - Default
  - Description
* - `model_identifier`
  - str
  - "sentence-transformers/all-MiniLM-L6-v2"
  - Pre-trained model for embedding generation
* - `embedding_model_inference_batch_size`
  - int
  - 256
  - Number of samples per embedding batch
* - `n_clusters`
  - int
  - 100
  - Number of clusters for k-means clustering
* - `kmeans_max_iter`
  - int
  - 300
  - Maximum iterations for clustering
* - `eps`
  - float
  - 0.01
  - Threshold for deduplication (higher = more aggressive)
* - `which_to_keep`
  - str
  - "hard"
  - Strategy for keeping duplicates ("hard"/"easy"/"random")
* - `pairwise_batch_size`
  - int
  - 1024
  - Batch size for similarity computation
* - `distance_metric`
  - str
  - "cosine"
  - Distance metric for similarity ("cosine" or "l2")
* - `embedding_pooling`
  - str
  - "mean_pooling"
  - Pooling strategy ("mean_pooling" or "last_token")
* - `perform_removal`
  - bool
  - true
  - Whether to perform duplicate removal
* - `text_field`
  - str
  - "text"
  - Name of the text field in input data
* - `id_field`
  - str
  - "_curator_dedup_id"
  - Name of the ID field in the data
```

### Similarity Threshold

Control deduplication aggressiveness with `eps`:

- **Lower values** (e.g., 0.001): More strict, less deduplication, higher confidence
- **Higher values** (e.g., 0.1): Less strict, more aggressive deduplication

Experiment with different values to balance data reduction and dataset diversity.

:::{dropdown} Embedding Models
:icon: beaker

**Sentence Transformers** (recommended for text):

```python
workflow = TextSemanticDeduplicationWorkflow(
    model_identifier="sentence-transformers/all-MiniLM-L6-v2",
    # ... other parameters
)
```

**HuggingFace Models**:

```python
workflow = TextSemanticDeduplicationWorkflow(
    model_identifier="facebook/opt-125m",
    # ... other parameters
)
```

**When choosing a model**:

- Ensure compatibility with your data type
- Adjust `embedding_model_inference_batch_size` for memory requirements
- Choose models appropriate for your language or domain
- Avoid generic decoder-only LLMs (e.g., OPT/GPT) for embeddings; prefer models trained for sentence embeddings (e.g., E5/BGE/SBERT)
:::

:::{dropdown} Advanced Configuration
:icon: gear

```python
workflow = TextSemanticDeduplicationWorkflow(
    # I/O
    input_path="input_data/",
    output_path="results/",
    cache_path="semdedup_cache",
    
    # Embedding generation
    text_field="text",
    model_identifier="sentence-transformers/all-MiniLM-L6-v2",
    embedding_max_seq_length=512,
    embedding_pooling="mean_pooling",
    embedding_model_inference_batch_size=256,
    
    # Deduplication
    n_clusters=100,
    eps=0.01,  # Similarity threshold
    distance_metric="cosine",
    which_to_keep="hard",
    
    # K-means
    kmeans_max_iter=300,
    kmeans_tol=1e-4,
    pairwise_batch_size=1024,
    
    perform_removal=True
)
```
:::

## Output Format

The semantic deduplication process produces the following directory structure in your configured `cache_path`:

```text
cache_path/
├── embeddings/                           # Embedding outputs
│   └── *.parquet                         # Parquet files containing document embeddings
├── semantic_dedup/                       # Semantic deduplication cache
│   ├── kmeans_results/                   # K-means clustering outputs
│   │   ├── kmeans_centroids.npy         # Cluster centroids
│   │   └── embs_by_nearest_center/      # Embeddings organized by cluster
│   │       └── nearest_cent={0..n-1}/   # Subdirectories for each cluster
│   │           └── *.parquet            # Cluster member embeddings
│   └── pairwise_results/                # Pairwise similarity results
│       └── *.parquet                    # Similarity scores by cluster
└── output_path/
    ├── duplicates/                       # Duplicate identification results
    │   └── *.parquet                    # Document IDs to remove
    └── deduplicated/                     # Final clean dataset (if perform_removal=True)
        └── *.parquet                    # Deduplicated documents
```

### File Formats

The workflow produces these output files:

1. **Document Embeddings** (`embeddings/*.parquet`):
   - Contains document IDs and their vector embeddings
   - Format: Parquet files with columns: `[id_column, embedding_column]`

2. **Cluster Assignments** (`semantic_dedup/kmeans_results/`):
   - `kmeans_centroids.npy`: NumPy array of cluster centers
   - `embs_by_nearest_center/`: Parquet files containing cluster members
   - Format: Parquet files with columns: `[id_column, embedding_column, cluster_id]`

3. **Duplicate IDs** (`output_path/duplicates/*.parquet`):
   - IDs of documents identified as duplicates for removal
   - Format: Parquet file with columns: `["id"]`
   - **Important**: Contains only the IDs of documents to remove, not the full document content
   - When `perform_removal=True`, clean dataset is saved to `output_path/deduplicated/`

 

:::{dropdown} Performance Considerations
:icon: zap

**Performance characteristics**:

- Computationally intensive, especially for large datasets
- GPU acceleration required for embedding generation and clustering
- Benefits often outweigh upfront cost (reduced training time, improved model performance)

**GPU requirements**:

- NVIDIA GPU with CUDA support
- Sufficient GPU memory (recommended: >8GB for medium datasets)
- RAPIDS libraries (cuDF) for GPU-accelerated dataframe operations
- CPU-only processing not supported

**Performance tuning**:

- Adjust `n_clusters` based on dataset size and available resources
- Use batched cosine similarity to reduce memory requirements
- Consider distributed processing for very large datasets

```{list-table} Performance Scaling (Example)
:header-rows: 1
:name: semantic-performance

* - Dataset Size
  - GPU Memory
  - Processing Time
  - Recommended GPUs
* - <100K docs
  - 4-8 GB
  - 1-2 hours
  - RTX 3080, A100
* - 100K-1M docs
  - 8-16 GB
  - 2-8 hours
  - RTX 4090, A100
* - >1M docs
  - >16 GB
  - 8+ hours
  - A100, H100
```

For more details, see the [SemDeDup paper](https://arxiv.org/pdf/2303.09540) by Abbas et al.
:::

:::{dropdown} Advanced Configuration
:icon: gear

**ID Generator for large-scale operations**:

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

# Use persisted ID generator in removal workflow
removal_workflow = TextDuplicatesRemovalWorkflow(
    input_path=input_path,
    ids_to_remove_path=duplicates_path,
    output_path=output_path,
    id_generator_path=id_generator_path,
    input_files_per_partition=1,  # Match partitioning as embedding generation
    # ... other parameters
)
```

**Critical requirements**:

- Use the same input configuration (file paths, partitioning) across all stages
- ID consistency maintained by hashing filenames in each task
- Mismatched partitioning causes ID lookup failures

**Ray backend configuration**:

```python
from nemo_curator.core.client import RayClient

client = RayClient(
    num_cpus=64,    # Adjust based on available cores
    num_gpus=4      # Should be roughly 2x the memory of embeddings
)
client.start()

try:
    workflow = TextSemanticDeduplicationWorkflow(
        input_path=input_path,
        output_path=output_path,
        cache_path=cache_path,
        # ... other parameters
    )
    results = workflow.run()
finally:
    client.stop()
```

Provides distributed processing, memory management, and fault tolerance.
:::
