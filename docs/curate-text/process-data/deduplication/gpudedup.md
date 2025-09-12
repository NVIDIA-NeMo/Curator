---
description: "Remove exact and fuzzy duplicates using hash-based algorithms with optional GPU acceleration and RAPIDS integration"
categories: ["how-to-guides"]
tags: ["hash-deduplication", "fuzzy-dedup", "exact-dedup", "minhash", "lsh", "rapids", "performance"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-format-gpu-dedup)=
# Hash-Based Duplicate Removal

Remove duplicate and near-duplicate documents from your text datasets using NeMo Curator's hash-based deduplication modules with optional GPU acceleration.

## How It Works

These modules use hash-based algorithms to efficiently process large datasets and support two primary methods: **Exact** and **Fuzzy Duplicate** removal. Fuzzy deduplication leverages [RAPIDS](https://rapids.ai) for GPU acceleration.

```{list-table} Hash-Based Duplicate Removal Methods
:header-rows: 1
:widths: 20 35 35 10

* - Method
  - Exact Duplicate Removal
  - Fuzzy Duplicate Removal
  - GPU Required
* - Purpose
  - Removes identical documents
  - Removes similar documents based on content
  - 
* - Process
  - 1. Hash document content
    2. Keep one document per unique hash
    3. Works on CPU or GPU (GPU recommended)
  - 1. Compute MinHash signatures
    2. Group via LSH buckets
    3. Optional similarity verification
    4. Keep one doc per similar group
  - Optional / Required
* - Best For
  - Finding exact copies
  - Finding near-duplicates and variants
  - 
```

Removing duplicates improves language model training by preventing overrepresentation of repeated content. For more information, see research by [Muennighoff et al. (2023)](https://arxiv.org/abs/2305.16264) and [Tirumala et al. (2023)](https://arxiv.org/abs/2308.12284).

<!-- Note: "overrepresentation", "Muennighoff", "Tirumala", and "Jaccard" are legitimate technical terms -->

---

## Understanding Operational Modes

The deduplication workflows support different operational modes:

```{list-table} Operational Modes
:header-rows: 1
:widths: 30 35 35

* - Workflow Type
  - `ExactDeduplicationWorkflow`
  - `FuzzyDeduplicationWorkflow`
* - Current Support
  - Identification only (`perform_removal=False`)
  - Identification only (`perform_removal=False`)
* - Output
  - Duplicate IDs written to output directory
  - Duplicate IDs written to output directory
* - Workflow
  - 1. Call `workflow.run()`
    2. Process output files separately for removal
  - 1. Call `workflow.run()`
    2. Process output files separately for removal
* - Use Case
  - Large-scale distributed exact duplicate identification
  - Large-scale distributed fuzzy duplicate identification
```

**Important Notes:**
- **Both workflows**: Currently support identification only. Removal functionality is planned for future releases
- **Output locations**: Both workflows write results to specified output directory, not as return values
- **Ray requirement**: Both workflows require Ray distributed computing framework with GPU support
- **ID assignment**: ExactDeduplicationWorkflow can automatically assign IDs or use existing ones

---

## Usage

### Exact Duplicate Removal

::::{tab-set}

:::{tab-item} Python
:sync: pyth-sync

```python
import ray
# Note: Import directly from workflow module (not available in __init__.py)
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

# Initialize Ray cluster (required for exact deduplication)
ray.init(num_gpus=4)

# Basic exact deduplication workflow
exact_workflow = ExactDeduplicationWorkflow(
    input_path="/path/to/input/data",
    output_path="/path/to/output",
    text_field="text",
    perform_removal=False,  # Currently only identification supported
    assign_id=True,         # Automatically assign unique IDs
    input_filetype="parquet",  # "parquet" or "jsonl"
    input_blocksize="2GiB"     # Block size for reading
)

# Run the workflow
exact_workflow.run()

# Advanced configuration with existing IDs and storage options
exact_workflow_advanced = ExactDeduplicationWorkflow(
    input_path="/path/to/input/data",
    output_path="/path/to/output",
    # Input configuration
    input_filetype="jsonl",
    input_blocksize="1GiB",
    input_file_extensions=[".jsonl"],
    # Storage options for remote filesystems
    read_kwargs={
        "storage_options": {
            "key": "your_access_key",
            "secret": "your_secret_key"
        }
    },
    write_kwargs={
        "storage_options": {
            "key": "your_access_key",
            "secret": "your_secret_key"
        }
    },
    # Processing configuration
    text_field="content",
    assign_id=False,        # Use existing ID field
    id_field="document_id", # Existing ID field name
    perform_removal=False,
    # Ray environment variables
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0,1,2,3"
    }
)

# Run with custom initial tasks (for integration with other pipelines)
from nemo_curator.tasks import FileGroupTask

initial_tasks = [
    FileGroupTask(
        task_id="batch_0",
        dataset_name="my_dataset",
        data=["/path/to/file1.parquet", "/path/to/file2.parquet"],
        _metadata={"source_files": ["/path/to/file1.parquet", "/path/to/file2.parquet"]}
    )
]
exact_workflow.run(initial_tasks=initial_tasks)

# Cleanup Ray when done
ray.shutdown()
```

**Performance Recommendations:**
- Uses MD5 hashing for exact duplicate detection
- Requires Ray cluster with GPU support
- Automatically partitions output to 1/3 the number of input tasks for efficiency
- Clear output directory between runs to avoid conflicts

**Output Structure:**
- **Output directory**: Contains duplicate IDs to remove and ID generator mapping
  - `ExactDuplicateIds/`: Parquet files with document IDs to remove
  - `exact_id_generator.json`: ID generator mapping (when `assign_id=True`)

**Workflow Stages:**
1. **File Partitioning**: Groups input files for parallel processing (if needed)
2. **Exact Duplicate Identification**: Computes MD5 hashes and identifies duplicates

**Ray Cluster Setup:**

The ExactDeduplicationWorkflow requires a Ray cluster with GPU support. Set up Ray before running:

```python
import ray

# Initialize Ray cluster (local with GPUs)
ray.init(num_gpus=4)

# Or connect to existing Ray cluster
# ray.init(address="ray://head-node-ip:10001")

# Run exact deduplication workflow
exact_workflow = ExactDeduplicationWorkflow(...)
exact_workflow.run()

# Shutdown Ray when done
ray.shutdown()
```

For distributed clusters, refer to the [Ray documentation](https://docs.ray.io/en/latest/cluster/getting-started.html) for cluster setup.
:::

:::{tab-item} CLI
:sync: cli-sync

```bash
# Add IDs if needed
add_id \
  --id-field-name="my_id" \
  --input-data-dir=/path/to/data \
  --id-prefix="doc_prefix"

# Remove exact duplicates
gpu_exact_dups \
  --input-data-dirs /path/to/jsonl/dir1 /path/to/jsonl/dir2 \
  --output-dir /path/to/output_dir \
  --input-json-text-field text \
  --input-json-id-field my_id \
  --log-dir ./
```

The CLI utilities only work with JSONL datasets and GPU-based backends. For other formats, use the Python API.

**Note**: CLI commands require proper installation of NeMo Curator with console script entry points. If CLI commands are not available, ensure you have installed the package correctly or use the Python API instead.
:::

::::

### Fuzzy Duplicate Removal

::::{tab-set}

:::{tab-item} Python
:sync: pyth-sync

```python
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

# Basic fuzzy deduplication workflow
fuzzy_workflow = FuzzyDeduplicationWorkflow(
    input_path="/path/to/input/data",
    cache_path="/path/to/cache",
    output_path="/path/to/output",
    text_field="text",
    perform_removal=False,  # Currently only duplicate identification is supported
    # MinHash parameters
    seed=42,
    char_ngrams=24,  # Character n-gram size for MinHash
    # LSH parameters  
    num_bands=20,           # Number of LSH bands
    minhashes_per_band=13,  # Hashes per band (affects similarity threshold)
    use_64_bit_hash=False,  # Use 32-bit or 64-bit hashes
    # Performance tuning
    bands_per_iteration=5,  # Bands processed concurrently
)

# Run the workflow (requires Ray cluster)
fuzzy_workflow.run()

# Advanced configuration with file format options
fuzzy_workflow_advanced = FuzzyDeduplicationWorkflow(
    input_path="/path/to/input/data",
    cache_path="/path/to/cache", 
    output_path="/path/to/output",
    # Input configuration
    input_filetype="parquet",  # "parquet" or "jsonl"
    input_blocksize="1GiB",    # Block size for reading
    input_file_extensions=[".parquet"],  # Optional: override default extensions
    # Storage options for remote filesystems
    read_kwargs={
        "storage_options": {
            "key": "your_access_key",
            "secret": "your_secret_key"
        }
    },
    cache_kwargs={
        "storage_options": {
            "key": "your_access_key", 
            "secret": "your_secret_key"
        }
    },
    write_kwargs={
        "storage_options": {
            "key": "your_access_key",
            "secret": "your_secret_key"
        }
    },
    # Processing configuration
    text_field="content",
    perform_removal=False,
    # MinHash + LSH tuning
    seed=123,
    char_ngrams=20,         # Smaller values may increase false positives
    num_bands=25,           # More bands = higher precision, lower recall
    minhashes_per_band=10,  # Fewer hashes per band = lower precision, higher recall
    use_64_bit_hash=True,   # Better for very large datasets
    bands_per_iteration=3,  # Reduce for memory-constrained environments
    # Ray environment variables
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0,1,2,3"
    }
)

# Run with custom initial tasks (for integration with other pipelines)
from nemo_curator.tasks import FileGroupTask

initial_tasks = [
    FileGroupTask(
        task_id="batch_0",
        dataset_name="my_dataset", 
        data=["/path/to/file1.parquet", "/path/to/file2.parquet"],
        _metadata={"source_files": ["/path/to/file1.parquet", "/path/to/file2.parquet"]}
    )
]
fuzzy_workflow.run(initial_tasks=initial_tasks)
```

**Performance Recommendations:**

- Use `char_ngrams >= 20` to minimize false positives (~5% with smaller values)
- Default parameters target approximately 0.8 Jaccard similarity
- Adjust `bands_per_iteration` based on available GPU memory
- Clear cache directory between runs to avoid conflicts
- Requires Ray cluster with GPU support

**Output Structure:**

- **Cache directory**: Contains intermediate files (MinHash signatures, LSH buckets, edges, connected components)
- **Output directory**: Contains duplicate IDs to remove and ID generator mapping
  - `FuzzyDuplicateIds/`: Parquet files with document IDs to remove
  - `fuzzy_id_generator.json`: ID generator mapping for future removal operations

**Workflow Stages:**

1. **File Partitioning**: Groups input files for parallel processing
2. **MinHash**: Computes MinHash signatures for each document
3. **LSH**: Performs Locality Sensitive Hashing to find candidate pairs
4. **Buckets to Edges**: Converts LSH buckets to graph edges
5. **Connected Components**: Finds connected components in the similarity graph
6. **Identify Duplicates**: Selects documents to remove from each duplicate group

**Ray Cluster Setup:**

The FuzzyDeduplicationWorkflow requires a Ray cluster with GPU support. Set up Ray before running:

```python
import ray

# Initialize Ray cluster (local with GPUs)
ray.init(num_gpus=4)

# Or connect to existing Ray cluster
# ray.init(address="ray://head-node-ip:10001")

# Run fuzzy deduplication workflow
fuzzy_workflow = FuzzyDeduplicationWorkflow(...)
fuzzy_workflow.run()

# Shutdown Ray when done
ray.shutdown()
```

For distributed clusters, refer to the [Ray documentation](https://docs.ray.io/en/latest/cluster/getting-started.html) for cluster setup.
:::

:::{tab-item} CLI
:sync: cli-sync

Fuzzy duplicate removal via the CLI involves several sequential steps:

```bash
# 1. Compute MinHash signatures
gpu_compute_minhashes \
  --input-data-dirs /path/to/jsonl/dir \
  --output-minhash-dir /path/to/output_minhashes \
  --input-json-text-field text \
  --input-json-id-field my_id \
  --minhash-length 256 \
  --char-ngram 24 \
  --seed 42

# 2. Generate LSH buckets
minhash_buckets \
  --input-data-dirs /path/to/output_minhashes \
  --output-bucket-dir /path/to/dedup_output \
  --input-minhash-field _minhash_signature \
  --input-json-id-field my_id \
  --num-bands 20

# 3. Generate edges from buckets
buckets_to_edges \
  --input-bucket-dir /path/to/dedup_output/_buckets.parquet \
  --output-dir /path/to/dedup_output \
  --input-json-id-field my_id

# 4. Find connected components
gpu_connected_component \
  --jaccard-pairs-path /path/to/dedup_output/_edges.parquet \
  --output-dir /path/to/dedup_output \
  --cache-dir /path/to/cc_cache \
  --input-json-id-field my_id
```

For more advanced configurations including similarity verification, refer to the full documentation.
:::

::::

### Incremental Processing

For new data additions, you don't need to reprocess existing documents:

1. Organize new data in separate directories
2. Compute MinHash signatures only for new data
3. Run subsequent steps on all data (existing and new MinHash signatures)

```bash
gpu_compute_minhashes \
  --input-data-dirs /input/new_data \
  --output-minhash-dir /output/ \
  --input-json-text-field text \
  --input-json-id-field my_id \
  --minhash-length 256 \
  --char-ngram 24
```

Then proceed with the remaining steps as usual on the combined MinHash directories. 

## Performance and GPU Requirements

### GPU Acceleration Overview

- **Exact Deduplication**:
  - **Backend Support**: Ray cluster with GPU support required
- **GPU Benefits**: Essential for MD5 hashing operations at scale
  - **Memory**: Distributed across Ray cluster nodes for large datasets

- **Fuzzy Deduplication**:
  - **Backend Support**: Ray cluster with GPU support required
  - **GPU Benefits**: Essential for MinHash and LSH operations at scale
  - **Memory**: Distributed across Ray cluster nodes for large datasets

### Performance Characteristics

```{list-table} Performance Comparison
:header-rows: 1
:widths: 25 25 25 25

* - Method
  - Small Datasets (<100K docs)
  - Medium Datasets (100K-1M docs)
  - Large Datasets (>1M docs)
* - Exact (Ray + GPU)
  - Fast
  - Fast
  - Very Fast
* - Fuzzy (Ray + GPU)
  - Fast
  - Fast
  - Very Fast
```

### Hardware Recommendations

- **CPU-only environments**: No deduplication workflows available (all require Ray + GPU)
- **Ray + GPU environments**: Both exact and fuzzy deduplication workflows require distributed Ray cluster with GPU support
- **Memory considerations**: GPU memory distributed across Ray cluster nodes for large datasets
- **Distributed processing**: Ray clusters required for all deduplication workflows

### Error Handling and Validation

When working with deduplication workflows, consider these common scenarios:

```python
import ray
import os
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

# Ensure Ray is initialized before deduplication
if not ray.is_initialized():
    ray.init(num_gpus=4)

# Handle output directory issues
output_path = "/path/to/output"
if os.path.exists(output_path):
    print(f"Warning: Output directory {output_path} exists and will be reused")
    # Clear if needed: shutil.rmtree(output_path)

# Handle ID generator conflicts for exact deduplication
try:
    exact_workflow = ExactDeduplicationWorkflow(
        input_path="/path/to/data",
        output_path=output_path,
        assign_id=True  # This requires ID generator
    )
    exact_workflow.run()
except RuntimeError as e:
    if "existing id generator actor" in str(e):
        print("ID generator actor already exists. Clean up previous run.")
        # Follow error message instructions to clean up
    else:
        raise

# Handle fuzzy deduplication with cache directory
try:
    cache_path = "/path/to/cache"
    fuzzy_workflow = FuzzyDeduplicationWorkflow(
        input_path="/path/to/data",
        cache_path=cache_path,
        output_path=output_path
    )
    fuzzy_workflow.run()
except RuntimeError as e:
    if "existing id generator actor" in str(e):
        print("ID generator actor already exists. Clean up previous run.")
    else:
        raise

# Check for no duplicates found
exact_output_path = f"{output_path}/ExactDuplicateIds"
fuzzy_output_path = f"{output_path}/FuzzyDuplicateIds"

if os.path.exists(exact_output_path) and len(os.listdir(exact_output_path)) == 0:
    print("No exact duplicates found in the dataset")

if os.path.exists(fuzzy_output_path) and len(os.listdir(fuzzy_output_path)) == 0:
    print("No fuzzy duplicates found in the dataset")

# Cleanup Ray when done
ray.shutdown()
```