---
description: "Identify near-duplicate documents using MinHash and LSH with GPU acceleration"
categories: ["how-to-guides"]
tags: ["fuzzy-dedup", "minhash", "lsh", "gpu", "ray"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-dedup-fuzzy)=

# Fuzzy Duplicate Removal

Find near-duplicate documents with small edits or reformatting using MinHash and Locality Sensitive Hashing (LSH). This approach identifies candidate pairs with a similarity threshold efficiently at scale on GPU.

For other approaches, refer to {ref}`Deduplication <text-process-data-dedup>`.

---

## How It Works

1. File partitioning for scalable, distributed processing
2. MinHash signatures over character n-grams
3. LSH banding to find candidate matches
4. Graph construction and connected components
5. Select one document per duplicate group and emit IDs to remove

---

## Usage

```python
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

# Basic fuzzy duplicate identification
fuzzy_workflow = FuzzyDeduplicationWorkflow(
    input_path="/path/to/input/data",
    cache_path="/path/to/cache",
    output_path="/path/to/output",
    text_field="text",
    perform_removal=False,  # Identification only
    input_blocksize="1GiB",  # Default block size for fuzzy dedup
    # MinHash parameters
    seed=42,
    char_ngrams=24,  # Character n-gram size for MinHash
    # LSH parameters
    num_bands=20,
    minhashes_per_band=13,
    use_64_bit_hash=False,
    # Performance tuning
    bands_per_iteration=5,
)

fuzzy_workflow.run()

# Advanced configuration (I/O and storage options)
fuzzy_workflow_advanced = FuzzyDeduplicationWorkflow(
    input_path="/path/to/input/data",
    cache_path="/path/to/cache",
    output_path="/path/to/output",
    input_filetype="parquet",   # "parquet" or "jsonl"
    input_blocksize="1GiB",
    input_file_extensions=[".parquet"],
    read_kwargs={"storage_options": {"key": "<access_key>", "secret": "<secret_key>"}},
    cache_kwargs={"storage_options": {"key": "<access_key>", "secret": "<secret_key>"}},
    write_kwargs={"storage_options": {"key": "<access_key>", "secret": "<secret_key>"}},
    text_field="content",
    perform_removal=False,
    seed=123,
    char_ngrams=20,
    num_bands=25,
    minhashes_per_band=10,
    use_64_bit_hash=True,
    bands_per_iteration=3,
    env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
)

fuzzy_workflow_advanced.run()
```

```{note}
Removal is currently not implemented in the fuzzy workflow (`perform_removal=True` raises an error). Use the duplicate ID outputs with the Text Duplicates Removal workflow. Refer to {ref}`Common Operations <text-process-data-dedup-common-ops>` for removal and outputs.
```

---

## Performance Recommendations

- Use `char_ngrams >= 20` to reduce false positives
- Adjust `bands_per_iteration` based on available GPU memory
- Requires a Ray-based distributed GPU execution environment
- Clear the cache and output directories between runs to avoid conflicts

---

## Output Structure

- Cache directory:
  - `MinHashStage/`, `LSHStage/`, `BucketsToEdges/`, `ConnectedComponents/`
- Output directory:
  - `FuzzyDuplicateIds/`: Parquet files with document IDs to remove
  - `fuzzy_id_generator.json`: ID generator mapping

---

## Workflow Stages

1. File Partitioning
2. MinHash
3. LSH
4. Buckets to Edges
5. Connected Components
6. Identify Duplicates
