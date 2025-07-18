---
description: "Core concepts for loading and managing text datasets including DocumentDataset, ParallelDataset, and supported file formats"
categories: ["concepts-architecture"]
tags: ["data-loading", "document-dataset", "parallel-dataset", "distributed", "gpu-accelerated", "local-files"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "text-only"
---

(about-concepts-text-data-loading)=
# Data Loading Concepts

This guide covers the core concepts for loading and managing text data from local files in NVIDIA NeMo Curator.

(documentdataset)=
## DocumentDataset

`DocumentDataset` is the foundation for handling large-scale text data processing in NeMo Curator. It is built on top of `Dask` DataFrames (`dd.DataFrame`) to enable distributed processing of local files.

```{list-table}
:header-rows: 1

* - Feature
  - Description
* - Lazy Loading & Memory Management
  - - On-demand data loading through `Dask`
    - Automatic memory optimization via `Dask's` distributed computing
    - Support for both CPU and GPU memory
    - Partition-based processing for scalability
* - GPU Acceleration
  - - Seamless CPU/GPU memory movement
    - GPU-accelerated operations via cuDF backend (optional)
    - Integration with RAPIDS ecosystem
    - Configurable through backend parameter
* - Robust Processing
  - - State tracking for interrupted operations
    - Recovery mechanisms through `Dask` persistence
    - Distributed processing support
    - Automatic partition management
```

:::{dropdown} Usage Examples
:icon: code-square

```python
# Creating DocumentDataset from different sources
from nemo_curator.datasets import DocumentDataset

# Read JSONL files
dataset = DocumentDataset.read_json("data.jsonl")

# Read Parquet files with GPU acceleration
gpu_dataset = DocumentDataset.read_parquet(
    "data.parquet",
    backend="cudf"  # Enable GPU acceleration
)

# Read multiple files
dataset = DocumentDataset.read_json([
    "data1.jsonl",
    "data2.jsonl"
])

# Basic operations
print(f"Dataset size: {len(dataset)}")
sample_data = dataset.head(10)  # Get first 10 rows
persisted = dataset.persist()   # Persist in memory
repartitioned = dataset.repartition(npartitions=4)  # Repartition

# Convert to pandas for local processing
pandas_df = dataset.to_pandas()
```

:::

## ParallelDataset

`ParallelDataset` extends `DocumentDataset` to handle parallel text data, particularly for machine translation and cross-lingual tasks.

```{list-table}
:header-rows: 1

* - Feature
  - Description
* - Parallel Text Processing
  - - Line-aligned file handling
    - Language pair management
    - Document ID tracking
    - Format conversion
    - Built-in length ratio filtering
* - Quality Filters
  - - Length ratio validation
    - Language identification
    - Parallel text scoring
    - Custom bitext filters
* - Output Formats
  - - Aligned text files
    - JSONL/Parquet export
    - Distributed writing support
    - Language-specific file naming
```

:::{dropdown} Usage Examples
:icon: code-square

```python
# Loading parallel text files (single pair)
from nemo_curator.datasets import ParallelDataset

dataset = ParallelDataset.read_simple_bitext(
    src_input_files="data.en",
    tgt_input_files="data.de",
    src_lang="en",
    tgt_lang="de"
)

# Multiple file pairs
dataset = ParallelDataset.read_simple_bitext(
    src_input_files=["train.en", "dev.en"],
    tgt_input_files=["train.de", "dev.de"],
    src_lang="en",
    tgt_lang="de"
)

# Apply length ratio filter
from nemo_curator.filters import LengthRatioFilter
length_filter = LengthRatioFilter(max_ratio=3.0)
filtered_dataset = length_filter(dataset)

# Export processed data
dataset.to_bitext(
    output_file_dir="processed_data/",
    write_to_filename=True
)
```
:::

(data-loading-file-formats)=
## Supported File Formats

DocumentDataset supports multiple file formats for loading text data from local files:

::::{tab-set}

:::{tab-item} JSONL
:sync: `jsonl`

**JSON Lines format** - Most commonly used format for text datasets in NeMo Curator.

```python
# Single file
dataset = DocumentDataset.read_json("data.jsonl")

# Multiple files
dataset = DocumentDataset.read_json([
    "file1.jsonl", 
    "file2.jsonl"
])

# Directory of files
dataset = DocumentDataset.read_json("data_directory/")

# Performance optimization with column selection
dataset = DocumentDataset.read_json(
    "data.jsonl", 
    columns=["text", "id"]
)
```

{bdg-secondary}`most-common` {bdg-secondary}`fast-loading`

:::

:::{tab-item} Parquet
:sync: parquet

**Columnar format** - Better performance for large datasets and GPU acceleration.

```python
# Basic Parquet reading
dataset = DocumentDataset.read_parquet("data.parquet")

# GPU acceleration (recommended for production)
dataset = DocumentDataset.read_parquet(
    "data.parquet",
    backend="cudf"
)

# Column selection for better performance
dataset = DocumentDataset.read_parquet(
    "data.parquet",
    columns=["text", "metadata"]
)
```

{bdg-secondary}`production` {bdg-secondary}`gpu-optimized`

:::

:::{tab-item} Pickle
:sync: pickle

**Python serialization** - For preserving complex data structures.

```python
# Read pickle files
dataset = DocumentDataset.read_pickle("data.pkl")

# Multiple pickle files
dataset = DocumentDataset.read_pickle([
    "data1.pkl",
    "data2.pkl"
])
```

{bdg-secondary}`python-native` {bdg-secondary}`object-preservation`

:::

:::{tab-item} Custom
:sync: custom

**Custom formats** - Extensible framework for specialized file readers.

```python
# Custom file format
dataset = DocumentDataset.read_custom(
    input_files="custom_data.ext",
    file_type="ext",
    read_func_single_partition=my_custom_reader,
    backend="pandas"
)
```

{bdg-secondary}`extensible` {bdg-secondary}`specialized`

:::

::::

## Data Export Options

NeMo Curator provides flexible export options for processed datasets:

::::{tab-set}

:::{tab-item} JSONL Export
:sync: `jsonl`-export

**JSON Lines export** - Human-readable format for text datasets.

```python
# Basic export
dataset.to_json("output_directory/")

# Export with filename preservation
dataset.to_json(
    "output_directory/",
    write_to_filename=True,
    keep_filename_column=True
)

# Partitioned export
dataset.to_json(
    "output_directory/",
    partition_on="language"
)
```

{bdg-secondary}`human-readable` {bdg-secondary}`debugging-friendly`

:::

:::{tab-item} Parquet Export
:sync: parquet-export

**Parquet export** - Optimized columnar format for production workflows.

```python
# Basic export
dataset.to_parquet("output_directory/")

# Export with partitioning
dataset.to_parquet(
    "output_directory/",
    partition_on="category"
)

# GPU-accelerated export
dataset.to_parquet(
    "output_directory/",
    backend="cudf"
)
```

{bdg-secondary}`high-performance` {bdg-secondary}`production-ready`

:::

::::

## Common Loading Patterns

::::{tab-set}

:::{tab-item} Multiple Sources
:sync: multiple-sources

**Loading from multiple sources** - Combine data from different locations and formats.

```python
# Combine multiple directories
dataset = DocumentDataset.read_json([
    "dataset_v1/",
    "dataset_v2/",
    "additional_data/"
])

# Mix file types (not recommended, convert to consistent format first)
jsonl_data = DocumentDataset.read_json("text_data.jsonl")
parquet_data = DocumentDataset.read_parquet("structured_data.parquet")

# Combine datasets after loading
combined = DocumentDataset.from_pandas(
    pd.concat([jsonl_data.to_pandas(), parquet_data.to_pandas()])
)
```

{bdg-secondary}`data-aggregation` {bdg-secondary}`multi-source`

:::

:::{tab-item} Performance Optimization
:sync: performance

**Performance optimization** - Maximize throughput and minimize memory usage.

```python
# Optimize for GPU processing
dataset = DocumentDataset.read_parquet(
    "large_dataset.parquet",
    backend="cudf",
    columns=["text", "id"],  # Only load needed columns
    files_per_partition=4    # Optimize partition size
)

# Optimize memory usage
dataset = DocumentDataset.read_json(
    "data.jsonl",
    blocksize="512MB",  # Adjust based on available memory
    backend="pandas"
)

# Parallel loading with custom partition size
dataset = DocumentDataset.read_parquet(
    "data/",
    npartitions=16  # Match CPU/GPU count
)
```

{bdg-secondary}`high-performance` {bdg-secondary}`memory-efficient`

:::

:::{tab-item} Large Datasets
:sync: large-datasets

**Working with large datasets** - Handle massive datasets efficiently.

```python
# Efficient processing for large datasets
dataset = DocumentDataset.read_parquet("massive_dataset/")

# Persist in memory for repeated operations
dataset = dataset.persist()

# Repartition for optimal processing
dataset = dataset.repartition(npartitions=8)

# Process in chunks
for partition in dataset.to_delayed():
    # Process each partition separately
    result = partition.compute()

# Lazy evaluation for memory efficiency
dataset = dataset.map_partitions(
    lambda df: df.head(1000)  # Process only first 1000 rows per partition
)
```

{bdg-secondary}`scalable` {bdg-secondary}`memory-conscious`

:::

::::

## Remote Data Acquisition

For users who need to download and process data from remote sources, NeMo Curator provides a comprehensive data acquisition framework. This is covered in detail in {ref}`Data Acquisition Concepts <about-concepts-text-data-acquisition>`, which includes:

- **DocumentDownloader, DocumentIterator, DocumentExtractor** components
- **Built-in support** for Common Crawl, ArXiv, Wikipedia, and custom sources  
- **Integration patterns** with DocumentDataset
- **Configuration and scaling** strategies

The data acquisition process produces standard `DocumentDataset` objects that integrate seamlessly with the local file loading concepts covered on this page.