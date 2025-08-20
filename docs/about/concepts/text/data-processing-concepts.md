---
description: "Text processing workflows including quality filtering, fuzzy deduplication, content cleaning, and pipeline design"
categories: ["concepts-architecture"]
tags: ["data-processing", "quality-filtering", "deduplication", "pipeline", "pii-removal", "distributed"]
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
- `RepeatedLinesFilter` - Remove repetitive content
- `PunctuationFilter` - Ensure proper sentence structure
- `BoilerPlateStringFilter` - Remove template/boilerplate text

### 2. Fuzzy Deduplication 

For production datasets, fuzzy deduplication is essential to remove near-duplicate content across sources:

**Key Components:**
- `FuzzyDuplicates` - Main deduplication engine
- `FuzzyDuplicatesConfig` - Configuration for LSH parameters
- Connected components clustering for duplicate identification

### 3. Content Cleaning 

Basic text normalization and cleaning operations:

**Common Cleaning Steps:**
- `UnicodeReformatter` - Normalize Unicode characters
- `PiiModifier` - Remove or redact personal information
- `NewlineNormalizer` - Standardize line breaks
- Basic HTML/markup removal

### 4. Exact Deduplication 

Remove identical documents, especially useful for smaller datasets:

**Implementation:**
- `ExactDuplicates` - Hash-based exact matching
- MD5 or SHA-256 hashing for document identification

## Core Processing Architecture

NeMo Curator uses these fundamental building blocks that users combine into pipelines:

```{list-table}
:header-rows: 1

* - Component
  - Purpose  
  - Usage Pattern
* - **`DocumentDataset`**
  - Load, process, and save text data
  - Every workflow starts here
* - **`get_client()`**
  - Initialize distributed processing
  - Required for all workflows
* - **`ScoreFilter`**
  - Apply filters with optional scoring
  - Chain multiple quality filters
* - **`Sequential`**
  - Combine processing steps
  - Build multi-stage pipelines  
* - **`Modify`**
  - Transform document content
  - Clean and normalize text
```

## Implementation Examples

### Complete Quality Filtering Pipeline

This is the most common starting workflow, used in 90% of production pipelines:

```python
import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import (
    WordCountFilter,
    NonAlphaNumericFilter,
    RepeatedLinesFilter,
    PunctuationFilter,
    BoilerPlateStringFilter
)
from nemo_curator.utils.distributed_utils import get_client

# Initialize distributed processing (required for all workflows)
client = get_client()  # Defaults to CPU cluster - use cluster_type="gpu" for acceleration

# Load dataset - the starting point for all workflows
dataset = DocumentDataset.read_json("data/*.jsonl")

# Standard quality filtering pipeline (most common)
quality_filters = nc.Sequential([
    # Remove too short/long documents (essential)
    nc.ScoreFilter(
        WordCountFilter(min_words=50, max_words=10000),
        text_field="text",
        score_field="word_count"
    ),
    # Remove symbol-heavy content
    nc.ScoreFilter(
        NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
        text_field="text"
    ),
    # Remove repetitive content
    nc.ScoreFilter(
        RepeatedLinesFilter(max_repeated_line_fraction=0.7),
        text_field="text"
    ),
    # Ensure proper sentence structure
    nc.ScoreFilter(
        PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85),
        text_field="text"
    ),
    # Remove template/boilerplate text
    nc.ScoreFilter(
        BoilerPlateStringFilter(),
        text_field="text"
    )
])

# Apply filtering
filtered_dataset = quality_filters(dataset)
filtered_dataset.to_json("filtered_data/")
```

### Content Cleaning Pipeline

Basic text normalization:

```python
from nemo_curator.modifiers import UnicodeReformatter
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.utils.distributed_utils import get_client

# Initialize distributed processing
client = get_client()  # Use cluster_type="gpu" for faster processing when available

# Essential cleaning steps
cleaning_pipeline = nc.Sequential([
    # Normalize unicode characters (very common)
    nc.Modify(UnicodeReformatter()),
    # Remove/redact PII (important for production)
    nc.Modify(PiiModifier(
        supported_entities=["PERSON", "EMAIL", "PHONE_NUMBER"],
        anonymize_action="replace"
    ))
])

cleaned_dataset = cleaning_pipeline(dataset)
```

### Large-Scale Fuzzy Deduplication

Critical for production datasets (requires GPU):

```python
from nemo_curator import FuzzyDuplicates, FuzzyDuplicatesConfig
from nemo_curator.utils.distributed_utils import get_client

# Initialize GPU processing (required for fuzzy deduplication)
client = get_client(cluster_type="gpu")

# Configure fuzzy deduplication (production settings)
fuzzy_config = FuzzyDuplicatesConfig(
    cache_dir="./cache",
    hashes_per_bucket=13,  # LSH parameter
    num_bands=8,           # LSH bands for ~85% similarity threshold
    minhash_length=128     # Signature length
)

# Apply fuzzy deduplication
dedup_pipeline = FuzzyDuplicates(fuzzy_config)
deduplicated_dataset = dedup_pipeline(dataset)
```

### Exact Deduplication (All dataset sizes)

Quick deduplication for any dataset size:

```python
from nemo_curator.modules import ExactDuplicates
from nemo_curator.utils.distributed_utils import get_client

# Initialize distributed processing (works on CPU or GPU)
client = get_client()  # Use cluster_type="gpu" for faster hashing when available

# Remove exact duplicates using MD5 hashing
exact_dedup = ExactDuplicates(
    id_field="id", 
    text_field="text", 
    hash_method="md5"
)

# Find duplicates
duplicates = exact_dedup(dataset)
# Remove them
deduped_dataset = exact_dedup.remove(dataset, duplicates)
```

### Complete End-to-End Pipeline

Most users combine these steps into a comprehensive workflow:

```python
from nemo_curator.utils.distributed_utils import get_client

# Initialize distributed processing
client = get_client()  # Defaults to CPU - add cluster_type="gpu" for acceleration

# Complete production pipeline (most common pattern)
def build_production_pipeline():
    return nc.Sequential([
        # 1. Content cleaning first
        nc.Modify(UnicodeReformatter()),
        nc.Modify(PiiModifier(supported_entities=["PERSON"], anonymize_action="replace")),
        
        # 2. Quality filtering
        nc.ScoreFilter(WordCountFilter(min_words=50, max_words=10000), text_field="text"),
        nc.ScoreFilter(NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25), text_field="text"),
        nc.ScoreFilter(RepeatedLinesFilter(max_repeated_line_fraction=0.7), text_field="text"),
        nc.ScoreFilter(BoilerPlateStringFilter(), text_field="text"),
        
        # 3. Deduplication (fuzzy or exact depending on scale)
    ])

# Apply the complete pipeline
complete_pipeline = build_production_pipeline()
processed_dataset = complete_pipeline(dataset)

# Then apply deduplication separately for large datasets
if len(dataset) > 1_000_000:  # Large dataset
    fuzzy_dedup = FuzzyDuplicates(FuzzyDuplicatesConfig(cache_dir="./cache"))
    final_dataset = fuzzy_dedup(processed_dataset)
else:  # Smaller dataset
    exact_dedup = ExactDuplicates(id_field="id", text_field="text", hash_method="md5")
    duplicates = exact_dedup(processed_dataset)
    final_dataset = exact_dedup.remove(processed_dataset, duplicates)
```

## Advanced Usage Patterns

### GPU-Accelerated Processing

For faster processing when GPUs are available (some operations require GPU):

```python
from nemo_curator.utils.distributed_utils import get_client

# Initialize GPU cluster for acceleration
client = get_client(
    cluster_type="gpu",
    rmm_pool_size="4GB",
    enable_spilling=True
)

# Process dataset with GPU acceleration
dataset = DocumentDataset.read_json("data/*.jsonl", backend="cudf")

# Apply processing with GPU acceleration
processed_dataset = complete_pipeline(dataset)
```

**GPU acceleration benefits**:
- **Required** for fuzzy deduplication operations
- Faster processing for classification and embedding operations
- More efficient memory usage with RMM for large datasets
- Significant speedup for MinHash and LSH operations (16x faster for fuzzy deduplication)

### Multi-Node Distributed Processing

For production-scale data processing across multiple machines:

```python
from nemo_curator.utils.distributed_utils import get_client

# Connect to existing multi-node cluster
client = get_client(
    scheduler_address="tcp://scheduler-node:8786"
)

# Process large dataset across multiple nodes
large_dataset = DocumentDataset.read_json("large_data/*.jsonl", backend="cudf")

# Apply fuzzy deduplication at scale (most common large-scale operation)
fuzzy_config = FuzzyDuplicatesConfig(
    cache_dir="./cache",
    hashes_per_bucket=13,
    num_bands=8
)
fuzzy_dedup = FuzzyDuplicates(fuzzy_config)
deduplicated_large = fuzzy_dedup(large_dataset)

# Save results with partitioning for efficient storage
deduplicated_large.to_json("output/", write_to_filename=True)
```

### Domain-Specific Processing

Common patterns for specialized content:

```python
from nemo_curator.utils.distributed_utils import get_client

# Initialize distributed processing
client = get_client()  # Add cluster_type="gpu" for acceleration when available

# Web crawl data processing (very common)
web_pipeline = nc.Sequential([
    nc.ScoreFilter(WordCountFilter(min_words=100)),          # Web pages are longer
    nc.ScoreFilter(NonAlphaNumericFilter(max_ratio=0.3)),    # More lenient for web
    nc.ScoreFilter(BoilerPlateStringFilter()),               # Remove navigation/footers
    nc.ScoreFilter(UrlsFilter(max_url_ratio=0.2)),          # Limit URL-heavy content
])

# Code dataset processing
code_pipeline = nc.Sequential([
    nc.ScoreFilter(AlphaFilter(min_alpha_ratio=0.25)),       # Code has symbols
    nc.ScoreFilter(TokenCountFilter(min_tokens=20)),         # Reasonable file sizes
    nc.ScoreFilter(PythonCommentToCodeFilter()),             # Code quality metrics
])

# Academic/research content
academic_pipeline = nc.Sequential([
    nc.ScoreFilter(WordCountFilter(min_words=500)),          # Academic papers are longer
    nc.ScoreFilter(FastTextQualityFilter(model="academic")), # Domain-specific quality
])
```

### Configuration-Driven Processing

For reproducible production pipelines:

```python
from nemo_curator.utils.distributed_utils import get_client

# Initialize distributed processing
client = get_client()  # Add cluster_type="gpu" for acceleration when available

# Most production users define pipelines in configuration
def build_config_pipeline(config_file):
    """Build pipeline from YAML configuration"""
    # Load and parse configuration
    filter_pipeline = build_filter(config_file)
    return filter_pipeline

# Use configuration for consistent processing
config_pipeline = build_config_pipeline("production_filters.yaml")
processed_data = config_pipeline(dataset)
```

## Performance Best Practices

### Scale-Based Approach Selection

```{list-table}
:header-rows: 1

* - Dataset Size
  - Recommended Approach
  - Key Considerations
* - **Small (<1GB)**
  - Single node, exact deduplication
  - CPU cluster suitable, GPU optional for speed
* - **Medium (1-100GB)**
  - Single node, fuzzy deduplication
  - GPU required for fuzzy deduplication operations  
* - **Large (>100GB)**
  - Multi-node cluster, optimized fuzzy dedup
  - Distributed processing with GPU acceleration
```

### Hardware-Based Recommendations

```{list-table}
:header-rows: 1

* - Available Hardware
  - Recommended Setup
  - Performance Benefits
* - **GPU Available**
  - `get_client(cluster_type="gpu")`
  - Required for fuzzy deduplication, faster classification and embeddings
* - **CPU Only**
  - `get_client()` (default)
  - Good performance for filtering and exact deduplication
* - **Multi-Node Cluster**
  - `get_client(scheduler_address="...")`
  - Scales to massive datasets, distributes compute across nodes
```

### Production Optimization Guidelines

```python
from nemo_curator.utils.distributed_utils import get_client

# Initialize distributed processing (choose based on operations needed)
client = get_client()  # CPU default - reliable for all basic operations

# 1. Order operations by computational cost (most important optimization)
production_pipeline = nc.Sequential([
    # Cheapest operations first (filter out bad data early)
    nc.ScoreFilter(WordCountFilter(min_words=10)),        # Very fast
    nc.ScoreFilter(NonAlphaNumericFilter()),              # Fast
    nc.ScoreFilter(RepeatedLinesFilter()),                # Medium cost
    
    # More expensive operations on remaining data
    nc.ScoreFilter(FastTextQualityFilter()),              # Benefits from GPU acceleration
    # Deduplication separate and last (most expensive)
])

# 2. Use appropriate backend for your operations
dataset = DocumentDataset.read_json("data/*.jsonl")  # pandas backend (CPU)
# For GPU operations, convert: dataset.df.to_backend("cudf")

# 3. Batch processing for memory efficiency
processed = production_pipeline(dataset)
processed.to_json("output/", files_per_partition=1)  # Control output partitioning
```

### Advanced Client Configuration

For specialized use cases, configure the client with specific parameters:

```python
# GPU acceleration for operations that support or require it
client = get_client(
    cluster_type="gpu",
    rmm_pool_size="8GB",
    enable_spilling=True,
    set_torch_to_use_rmm=True
)

# Multi-node production cluster
client = get_client(
    scheduler_address="tcp://scheduler-node:8786"
)

# Custom CPU cluster configuration
client = get_client(
    cluster_type="cpu",
    n_workers=16,
    threads_per_worker=2,
    memory_limit="8GB"
)
```

## Command Line Usage

Most production users prefer command-line tools for automation. All NeMo Curator scripts automatically set up distributed processing:

```bash
# Most common: Basic quality filtering (uses get_client internally)
filter_documents \
  --input-data-dir=input/ \
  --filter-config-file=heuristic_filters.yaml \
  --output-retained-document-dir=output/ \
  --device=cpu \
  --num-workers=8

# GPU acceleration for faster processing
filter_documents \
  --input-data-dir=input/ \
  --filter-config-file=heuristic_filters.yaml \
  --output-retained-document-dir=output/ \
  --device=gpu

# Large-scale: Fuzzy deduplication (4-step process)
# Step 1: Compute minhashes
gpu_compute_minhashes \
  --input-data-dir=input/ \
  --output-minhash-dir=minhashes/ \
  --cache-dir=cache/ \
  --device=gpu

# Step 2: LSH bucketing  
minhash_buckets \
  --input-minhash-dir=minhashes/ \
  --output-bucket-dir=buckets/ \
  --cache-dir=cache/

# Step 3: Find duplicate pairs
buckets_to_edges \
  --input-bucket-dir=buckets/ \
  --output-dir=edges/ \
  --cache-dir=cache/

# Step 4: Remove duplicates
gpu_connected_component \
  --input-edges-dir=edges/ \
  --output-dir=deduplicated/ \
  --cache-dir=cache/

# Multi-node processing using scheduler
filter_documents \
  --input-data-dir=input/ \
  --filter-config-file=heuristic_filters.yaml \
  --output-retained-document-dir=output/ \
  --scheduler-address=tcp://scheduler-node:8786
```

### Common Command Line Options

All NeMo Curator scripts support these distributed processing options:

- `--device`: Choose `cpu` or `gpu` for processing (default: `cpu`)
- `--num-workers`: Number of workers for local processing (default: CPU count)
- `--scheduler-address`: Connect to existing distributed cluster
- `--scheduler-file`: Path to Dask scheduler file
- `--threads-per-worker`: Threads per worker (default: `1`)

These options automatically configure `get_client()` with the appropriate parameters.
