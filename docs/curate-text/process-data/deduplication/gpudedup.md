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

Both `ExactDuplicates` and `FuzzyDuplicates` support two operational modes controlled by the `perform_removal` parameter:

```{list-table} Operational Modes
:header-rows: 1
:widths: 30 35 35

* - Mode
  - `perform_removal=False` (Default)
  - `perform_removal=True`
* - Return Value
  - Dataset with duplicate IDs/groups
  - Deduplicated dataset
* - Workflow
  - 1. Call `module(dataset)` or `module.identify_duplicates(dataset)`
    2. Call `module.remove(dataset, duplicates)` 
  - 1. Call `module(dataset)` 
    2. Returns final deduplicated dataset
* - Use Case
  - When you want to inspect duplicates first
  - When you want direct deduplication
```

**Important Notes:**
- Exact deduplication: Returns documents with `_hashes` field when `perform_removal=False`
- Fuzzy deduplication: Returns documents with `group` field when `perform_removal=False`  
- Always check if the result is `None` (no duplicates found) before calling `.remove()`

---

## Usage

### Exact Duplicate Removal

::::{tab-set}

:::{tab-item} Python
:sync: pyth-sync

```python
from nemo_curator import ExactDuplicates, AddId
from nemo_curator.datasets import DocumentDataset

# Add unique IDs if needed
add_id = AddId(id_field="my_id", id_prefix="doc_prefix")
dataset = DocumentDataset.read_json("input_file_path")
id_dataset = add_id(dataset)

# Set up duplicate removal
exact_duplicates = ExactDuplicates(
  id_field="my_id",
  text_field="text",
  hash_method="md5",  # Currently only "md5" is supported
  perform_removal=True,  # If True, returns deduplicated dataset; if False, returns duplicate IDs
  cache_dir="/path/to/dedup_outputs",
)

# Process the dataset
dataset = DocumentDataset.read_parquet(
    input_files="/path/to/parquet/data",
    backend="cudf",  # or "pandas" for CPU
)
deduplicated_dataset = exact_duplicates(dataset)

# Alternative workflow when perform_removal=False:
# exact_duplicates = ExactDuplicates(
#     id_field="my_id",
#     text_field="text", 
#     hash_method="md5",
#     perform_removal=False,  # Returns duplicate IDs only
#     cache_dir="/path/to/dedup_outputs",
# )
# duplicates = exact_duplicates(dataset)  # Get duplicate IDs
# if duplicates is not None:
#     deduplicated_dataset = exact_duplicates.remove(dataset, duplicates)  # Remove duplicates
# else:
#     print("No duplicates found")
#     deduplicated_dataset = dataset
```

For a complete example, see [examples/exact_deduplication.py](https://github.com/NVIDIA/NeMo-Curator/blob/main/examples/exact_deduplication.py).
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
:::

::::

### Fuzzy Duplicate Removal

::::{tab-set}

:::{tab-item} Python
:sync: pyth-sync

```python
from nemo_curator import FuzzyDuplicates, FuzzyDuplicatesConfig
from nemo_curator.datasets import DocumentDataset

# Configure the duplicate removal
config = FuzzyDuplicatesConfig(
    cache_dir="/path/to/dedup_outputs",
    id_field="my_id",
    text_field="text",
    perform_removal=True,  # If True, returns deduplicated dataset; if False, returns duplicate IDs
    seed=42,
    char_ngrams=24,
    num_buckets=20,
    hashes_per_bucket=13,
    use_64_bit_hash=False,  # Set to True for 64-bit hashes
    false_positive_check=False,  # Set to True for higher accuracy but slower processing
)

# Initialize and run
fuzzy_duplicates = FuzzyDuplicates(
    config=config,
    logger="./",  # Optional: path to log directory or existing logger
)
dataset = DocumentDataset.read_json(
    input_files="/path/to/jsonl/data",
    backend="cudf",  # Fuzzy deduplication requires cuDF backend
)
deduplicated_dataset = fuzzy_duplicates(dataset)

# Alternative workflow when perform_removal=False:
# config.perform_removal = False
# fuzzy_duplicates = FuzzyDuplicates(config=config)
# duplicates = fuzzy_duplicates.identify_duplicates(dataset)  # Get duplicate groups
# if duplicates is not None:
#     deduplicated_dataset = fuzzy_duplicates.remove(dataset, duplicates)  # Remove duplicates
# else:
#     print("No duplicates found")
#     deduplicated_dataset = dataset
```

For best performance:
- Set `false_positive_check=False` for faster processing (may have ~5% false positives)
- The default parameters target approximately 0.8 Jaccard similarity
- Use `buckets_per_shuffle=1` for memory-constrained environments
- Clear the cache directory between runs to avoid conflicts
- Use GPU backend (`backend="cudf"`) for optimal performance

For a complete example, see [examples/fuzzy_deduplication.py](https://github.com/NVIDIA/NeMo-Curator/blob/main/examples/fuzzy_deduplication.py).
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
  - **Backend Support**: Both CPU (`pandas`) and GPU (`cudf`) 
  - **GPU Benefits**: Significant speedup for large datasets through optimized hashing
  - **Recommendation**: Use GPU for datasets with >1M documents

- **Fuzzy Deduplication**:
  - **Backend Support**: GPU only (`cudf` required)
  - **GPU Benefits**: Essential for MinHash and LSH operations
  - **Memory**: Requires sufficient GPU memory for dataset processing

### Performance Characteristics

```{list-table} Performance Comparison
:header-rows: 1
:widths: 25 25 25 25

* - Method
  - Small Datasets (<100K docs)
  - Medium Datasets (100K-1M docs)
  - Large Datasets (>1M docs)
* - Exact (CPU)
  - Fast
  - Moderate
  - Slow
* - Exact (GPU)
  - Fast
  - Fast
  - Fast
* - Fuzzy (GPU)
  - Fast
  - Fast
  - Fast
```

### Hardware Recommendations

- **CPU-only environments**: Use exact deduplication with `backend="pandas"`
- **GPU environments**: Use both exact and fuzzy deduplication with `backend="cudf"`
- **Memory considerations**: GPU memory should be >2x the dataset size in memory
- **Distributed processing**: Use Dask for datasets that exceed single GPU memory

### Error Handling and Validation

When working with deduplication modules, consider these common scenarios:

```python
# Check for empty results
duplicates = exact_duplicates.identify_duplicates(dataset)
if duplicates is None or len(duplicates) == 0:
    print("No duplicates found")
    return dataset

# Validate backend compatibility
try:
    # Fuzzy deduplication requires cuDF backend
    fuzzy_duplicates = FuzzyDuplicates(config=config)
    result = fuzzy_duplicates(dataset)
except ValueError as e:
    print(f"Backend error: {e}")
    # Convert to cuDF backend if needed
    dataset = dataset.to_backend("cudf")
    result = fuzzy_duplicates(dataset)

# Handle cache directory issues
import os
if os.path.exists(config.cache_dir):
    print(f"Warning: Cache directory {config.cache_dir} exists and will be reused")
    # Clear if needed: shutil.rmtree(config.cache_dir)
``` 