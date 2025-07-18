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

Semantic deduplication is particularly effective for large, uncurated web-scale datasets, where it can remove up to 50% of the data with minimal performance impact. The technique uses embeddings to identify "semantic duplicates" - content pairs that convey similar meaning despite using different words.

:::{note}
**GPU Acceleration**: Semantic deduplication requires GPU acceleration for both embedding generation and clustering operations. This method uses the `cudf` backend and PyTorch models on GPU for optimal performance.
:::

## How It Works

The SemDeDup algorithm consists of the following main steps:

1. **Embedding Generation**: Each document is embedded using a pre-trained model
2. **Clustering**: The embeddings are grouped into k clusters using k-means clustering
3. **Similarity Computation**: Within each cluster, pairwise cosine similarities are computed
4. **Duplicate Identification**: Document pairs with cosine similarity above a threshold are considered semantic duplicates
5. **Duplicate Removal**: From each group of semantic duplicates within a cluster, one representative document is kept (typically the one with the lowest cosine similarity to the cluster centroid) and the rest are removed

:::{note}
 NeMo Curator implements methods based on the paper [SemDeDup: Data-efficient learning at web-scale through semantic deduplication](https://arxiv.org/pdf/2303.09540) by Abbas et al.
:::

## Before You Start

Before running semantic deduplication, ensure that each document in your dataset has a unique identifier. You can use the `AddId` module from NeMo Curator if needed:

```python
from nemo_curator import AddId
from nemo_curator.datasets import DocumentDataset

# Add unique IDs to documents
add_id = AddId(id_field="doc_id", id_field_type="int")
dataset_with_ids = add_id(dataset)
```

## SemDedup Interface

The `SemDedup` class provides a flexible interface similar to other deduplication modules in NeMo Curator:

### Constructor Parameters
- `config`: SemDedupConfig object containing embedding and clustering settings
- `input_column`: Column name containing text data (default: "text")
- `id_column`: Column name containing document IDs (default: "id")
- `perform_removal`: Boolean flag controlling return behavior (default: False)
- `logger`: Logger instance or path to log directory

### Usage Modes

**Mode 1: Two-step process (`perform_removal=False`)**
```python
# Step 1: Identify duplicates
duplicates = sem_dedup(dataset)
# Step 2: Remove duplicates manually
deduplicated_dataset = sem_dedup.remove(dataset, duplicates)
```

**Mode 2: One-step process (`perform_removal=True`)**
```python
# Returns deduplicated dataset directly
deduplicated_dataset = sem_dedup(dataset)
```

---

## Quick Start

```python
from nemo_curator import SemDedup, SemDedupConfig
from nemo_curator.datasets import DocumentDataset

# Load your dataset (requires cudf backend for GPU acceleration)
dataset = DocumentDataset.read_json("input_data/*.jsonl", backend="cudf")

# Configure semantic deduplication
config = SemDedupConfig(
    cache_dir="./sem_cache",
    embedding_model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    n_clusters=10000,
    eps_to_extract=0.07  # Similarity threshold
)

# Option 1: Two-step process (more control)
sem_dedup = SemDedup(
    config=config,
    id_column="doc_id",
    perform_removal=False  # Returns duplicate IDs
)
duplicates = sem_dedup(dataset)
deduplicated_dataset = sem_dedup.remove(dataset, duplicates)

# Option 2: One-step process (simpler)
sem_dedup_simple = SemDedup(
    config=config,
    id_column="doc_id", 
    perform_removal=True  # Returns deduplicated dataset directly
)
deduplicated_dataset = sem_dedup_simple(dataset)
```

---

## Configuration

Semantic deduplication in NeMo Curator can be configured using a YAML file. Here's an example `sem_dedup_config.yaml`:

```yaml
# Configuration file for semantic dedup
cache_dir: "semdedup_cache"
num_files: -1
profile_dir: null  # Optional directory for Dask profiling

# Embeddings configuration
embedding_model_name_or_path: "sentence-transformers/all-MiniLM-L6-v2"
embedding_batch_size: 128
embeddings_save_loc: "embeddings"
embedding_max_mem_gb: null  # Auto-detected: GPU memory - 4GB
embedding_pooling_strategy: "mean_pooling"
embedding_column: "embeddings"
write_embeddings_to_disk: true
write_to_filename: false

# Clustering configuration
max_iter: 100
n_clusters: 1000
clustering_save_loc: "clustering_results"
random_state: 1234
sim_metric: "cosine"
which_to_keep: "hard"
batched_cosine_similarity: 1024
sort_clusters: true
kmeans_with_cos_dist: false
clustering_input_partition_size: "2gb"

# Extract dedup configuration
eps_thresholds: [0.01, 0.001]  # List of thresholds to compute
eps_to_extract: 0.01
```

You can customize this configuration file to suit your specific needs and dataset characteristics.

:::{note}
**Configuration Parameters**: The above configuration shows the most commonly used parameters. For advanced use cases, additional parameters like `profile_dir` (for Dask profiling), `embedding_max_mem_gb` (to control GPU memory usage), and clustering optimization parameters are available. See the complete parameter table below for all options.
:::

### Embedding Models

You can choose alternative pre-trained models for embedding generation by modifying the `embedding_model_name_or_path` parameter in the configuration file.

::::{tab-set}

:::{tab-item} Sentence Transformer

Sentence transformers are ideal for text-based semantic similarity tasks. 

```yaml
embedding_model_name_or_path: "sentence-transformers/all-MiniLM-L6-v2"
```
:::

:::{tab-item} Model

```yaml
embedding_model_name_or_path: "facebook/opt-125m"
```

You can also use your own pre-trained custom models by specifying the path.
:::
::::

When changing the model, ensure that:

1. The model is compatible with the data type you're working with
2. You adjust the `embedding_batch_size` parameter for your model's memory requirements
3. The chosen model is appropriate for the language or domain of your dataset

### Deduplication Threshold

The semantic deduplication process is controlled by the similarity threshold parameter:

```yaml
eps_to_extract: 0.01
```

`eps_to_extract`: The similarity threshold used for extracting deduplicated data. This value determines how similar documents need to be to be considered duplicates. Lower values are more strict, requiring higher similarity for documents to be considered duplicates.

When choosing an appropriate threshold:

* Lower thresholds (for example, 0.001): More strict, resulting in less deduplication but higher confidence in the identified duplicates
* Higher thresholds (for example, 0.1): Less strict, leading to more aggressive deduplication but potentially removing documents that are only somewhat similar

We recommend experimenting with different threshold values to find the optimal balance between data reduction and maintaining dataset diversity and quality. The impact of this threshold can vary depending on the nature and size of your dataset.

## Usage

::::{tab-set}

:::{tab-item} SemDedup Class
You can use the SemDedup class to perform all steps:

```python
from nemo_curator import SemDedup, SemDedupConfig
import yaml

# Load configuration from YAML file
with open("sem_dedup_config.yaml", "r") as config_file:
    config_dict = yaml.safe_load(config_file)

# Create SemDedupConfig object
config = SemDedupConfig(**config_dict)

# Initialize SemDedup with the configuration
sem_dedup = SemDedup(
    config=config,
    input_column="text",
    id_column="doc_id",
    perform_removal=False,  # Two-step process
    logger="path/to/log/dir",
)

# Two-step semantic deduplication process
# Step 1: Identify duplicates (returns duplicate IDs)
duplicates = sem_dedup(dataset)

# Step 2: Remove duplicates from original dataset
deduplicated_dataset = sem_dedup.remove(dataset, duplicates)

# Alternative: One-step process
# sem_dedup_onestep = SemDedup(config=config, perform_removal=True)
# deduplicated_dataset = sem_dedup_onestep(dataset)
```

This approach allows for easy experimentation with different configurations and models without changing the core code.

```{tip}
**Flexible Interface**: The `SemDedup` class supports both one-step and two-step workflows:
- Use `perform_removal=True` for direct deduplication (returns clean dataset)
- Use `perform_removal=False` for manual control over the removal process (returns duplicate IDs, then call `.remove()`)

This interface matches the behavior of other deduplication modules in NeMo Curator.
```
:::

:::{tab-item} Individual Components
Embedding Creation:

```python
from nemo_curator import EmbeddingCreator

# Generate embeddings for each document
embedding_creator = EmbeddingCreator(
    embedding_model_name_or_path="path/to/pretrained/model",
    embedding_batch_size=128,
    embedding_output_dir="path/to/output/embeddings",
    input_column="text",
    logger="path/to/log/dir",
)
embeddings_dataset = embedding_creator(dataset)
```

Clustering:

```python
from nemo_curator import ClusteringModel

# Cluster the embeddings
clustering_model = ClusteringModel(
    id_column="doc_id",
    max_iter=100,
    n_clusters=50000,
    clustering_output_dir="path/to/output/clusters",
    logger="path/to/log/dir"
)
clustered_dataset = clustering_model(embeddings_dataset)
```

Semantic Deduplication:

```python
from nemo_curator import SemanticClusterLevelDedup

# Perform semantic deduplication
semantic_dedup = SemanticClusterLevelDedup(
    n_clusters=50000,
    emb_by_clust_dir="path/to/embeddings/by/cluster",
    id_column="doc_id",
    which_to_keep="hard",
    batched_cosine_similarity=1024,
    output_dir="path/to/output/deduped",
    logger="path/to/log/dir"
)
semantic_dedup.compute_semantic_match_dfs()
# Returns dataset containing unique document IDs after deduplication
unique_document_ids = semantic_dedup.extract_dedup_data(eps_to_extract=0.07)

# Note: When using individual components, you need to filter manually
# The SemDedup class handles this filtering automatically when perform_removal=True
kept_ids = unique_document_ids.df["doc_id"].compute()  
deduplicated_dataset = original_dataset.df[original_dataset.df["doc_id"].isin(kept_ids)]
```

:::

::::

--- 

### Comparison with Other Deduplication Methods

```{list-table} Deduplication Method Behavior Comparison
:header-rows: 1
:widths: 20 25 25 30

* - Method
  - Return Value Options
  - perform_removal Parameter
  - Workflow
* - ExactDuplicates
  - Duplicates or Clean Dataset
  - ✅ Available
  - One-step or two-step
* - FuzzyDuplicates
  - Duplicates or Clean Dataset  
  - ✅ Available
  - One-step or two-step
* - SemDedup
  - IDs to Keep Only
  - ❌ Not Available
  - Always requires filtering step
```

### Key Parameters

```{list-table} Key Configuration Parameters
:header-rows: 1
:widths: 25 15 20 40

* - Parameter
  - Type
  - Default
  - Description
* - `embedding_model_name_or_path`
  - str
  - "sentence-transformers/all-MiniLM-L6-v2"
  - Pre-trained model for embedding generation
* - `embedding_batch_size`
  - int
  - 128
  - Number of samples per embedding batch
* - `embedding_max_mem_gb`
  - int
  - null
  - Maximum GPU memory for embeddings (auto-detected if null)
* - `n_clusters`
  - int
  - 1000
  - Number of clusters for k-means clustering
* - `max_iter`
  - int
  - 100
  - Maximum iterations for clustering
* - `eps_to_extract`
  - float
  - 0.01
  - Threshold for deduplication (higher = more aggressive)
* - `eps_thresholds`
  - list
  - [0.01, 0.001]
  - List of similarity thresholds to compute
* - `which_to_keep`
  - str
  - "hard"
  - Strategy for keeping duplicates ("hard"/"easy"/"random")
* - `batched_cosine_similarity`
  - int
  - 1024
  - Batch size for similarity computation
* - `clustering_input_partition_size`
  - str
  - "2gb"
  - Size of data partition for KMeans
* - `sort_clusters`
  - bool
  - true
  - Whether to sort clusters during processing
* - `kmeans_with_cos_dist`
  - bool
  - false
  - Whether to use cosine distance for KMeans
* - `write_embeddings_to_disk`
  - bool
  - true
  - Whether to save embeddings to disk
* - `write_to_filename`
  - bool
  - false
  - Whether to save embeddings to same filename as input
```

## Output Format

The semantic deduplication process produces the following directory structure in your configured `cache_dir`:

```s
cache_dir/
├── embeddings/                           # Embedding outputs
│   └── *.parquet                         # Parquet files containing document embeddings
├── clustering_results/                   # Clustering outputs
│   ├── kmeans_centroids.npy             # Cluster centroids
│   ├── embs_by_nearest_center/          # Embeddings organized by cluster
│   │   └── nearest_cent={0..n-1}/       # Subdirectories for each cluster
│   │       └── *.parquet                # Cluster member embeddings
│   └── unique_ids_{eps}.parquet         # Final deduplicated document IDs
└── *.log                                # Process logs
```

### File Formats

1. **Document Embeddings** (`embeddings/*.parquet`):
   - Contains document IDs and their vector embeddings
   - Format: Parquet files with columns: `[id_column, embedding_column]`

2. **Cluster Assignments** (`clustering_results/`):
   - `kmeans_centroids.npy`: NumPy array of cluster centers
   - `embs_by_nearest_center/`: Parquet files containing cluster members
   - Format: Parquet files with columns: `[id_column, embedding_column, cluster_id]`

3. **Deduplicated Results** (`clustering_results/unique_ids_{eps}.parquet`):
   - Final output containing unique document IDs after deduplication
   - One file per deduplication threshold (`eps`) from `eps_thresholds`
   - Format: Parquet file with columns: `[id_column, "dist", "cluster"]`
   - **Important**: Contains only the IDs of documents to keep, not the full document content
   - Use these IDs to filter your original dataset to obtain the deduplicated content

Typically, semantic deduplication reduces dataset size by 20–50% while maintaining or improving model performance.

## Performance Considerations

Semantic deduplication is computationally intensive, especially for large datasets. However, the benefits in terms of reduced training time and improved model performance often outweigh the upfront cost:

- Use GPU acceleration for faster embedding generation and clustering
- Adjust the number of clusters (`n_clusters`) based on your dataset size and available resources
- The `eps_to_extract` parameter controls the trade-off between dataset size reduction and potential information loss
- Using batched cosine similarity significantly reduces memory requirements for large datasets

### GPU Requirements

**Hardware Prerequisites:**
- NVIDIA GPU with CUDA support
- Sufficient GPU memory (recommended: >8GB for medium datasets)
- RAPIDS libraries (cuDF, cuML) for GPU-accelerated operations

**Backend Requirements:**
- **Required**: `cudf` backend for GPU acceleration
- **Not supported**: CPU-only processing (use hash-based deduplication instead)

**Performance Characteristics:**
- **Embedding Generation**: GPU-accelerated using PyTorch models
- **Clustering**: GPU-accelerated k-means clustering
- **Similarity Computation**: Batched GPU operations for cosine similarity

```{list-table} Performance Scaling
:header-rows: 1
:widths: 25 25 25 25

* - Dataset Size
  - GPU Memory Required
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

For very large datasets, consider distributed processing across multiple GPUs or use incremental processing approaches.

For more details on the algorithm and its performance implications, refer to the original paper: [SemDeDup: Data-efficient learning at web-scale through semantic deduplication](https://arxiv.org/pdf/2303.09540) by Abbas et al.
