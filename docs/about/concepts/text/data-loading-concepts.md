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

## Pipeline-Based Data Loading

NeMo Curator uses a **pipeline-based architecture** for handling large-scale text data processing. Data flows through processing stages that transform tasks, enabling distributed processing of local files.

```{list-table}
:header-rows: 1

* - Component
  - Description
* - Reader Stages
  - - `JsonlReader` and `ParquetReader` for file input
    - File partitioning and distributed loading
    - Column selection and performance optimization
    - Support for cloud storage via storage options
* - Processing Stages
  - - Modular stages for filtering, transformation, and analysis
    - `DocumentBatch` tasks flow between stages
    - GPU acceleration support via PyArrow and cuDF
    - Distributed execution via Ray or other backends
* - Pipeline Orchestration
  - - Compose stages into end-to-end workflows
    - Automatic task scheduling and resource management
    - State tracking and error recovery
    - Flexible execution backends (Xenna, Ray, etc.)
```

:::{dropdown} Usage Examples
:icon: code-square

```python
# Pipeline-based data loading with reader stages
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna.executor import XennaExecutor
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader

# Create pipeline with JSONL reader
pipeline = Pipeline(name="data_processing")

# Read JSONL files
jsonl_reader = JsonlReader(
    file_paths="data.jsonl",
    files_per_partition=4,
    fields=["text", "id"]  # Column selection
)
pipeline.add_stage(jsonl_reader)

# Read Parquet files with PyArrow optimization
parquet_reader = ParquetReader(
    file_paths="data.parquet",
    files_per_partition=4,
    fields=["text", "metadata"],
    read_kwargs={
        "engine": "pyarrow",
        "dtype_backend": "pyarrow"
    }
)

# Execute pipeline
executor = XennaExecutor()
results = pipeline.run(executor)

# Access results as DocumentBatch tasks
for task in results:
    df = task.to_pandas()  # Convert to pandas
    print(f"Processed {task.num_items} documents")
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
from nemo_curator.stages.text.io.reader import JsonlReader

# Single file
reader = JsonlReader(file_paths="data.jsonl")

# Multiple files
reader = JsonlReader(file_paths=[
    "file1.jsonl", 
    "file2.jsonl"
])

# Directory of files
reader = JsonlReader(file_paths="data_directory/")

# Performance optimization with column selection
reader = JsonlReader(
    file_paths="data.jsonl", 
    fields=["text", "id"]
)
```

{bdg-secondary}`most-common` {bdg-secondary}`fast-loading`

:::

:::{tab-item} Parquet
:sync: parquet

**Columnar format** - Better performance for large datasets and PyArrow optimization.

```python
from nemo_curator.stages.text.io.reader import ParquetReader

# Basic Parquet reading
reader = ParquetReader(file_paths="data.parquet")

# PyArrow optimization (default, recommended for production)
reader = ParquetReader(
    file_paths="data.parquet",
    read_kwargs={
        "engine": "pyarrow",
        "dtype_backend": "pyarrow"
    }
)

# Column selection for better performance
reader = ParquetReader(
    file_paths="data.parquet",
    fields=["text", "metadata"]
)
```

{bdg-secondary}`production` {bdg-secondary}`gpu-optimized`

:::

:::{tab-item} Pickle
:sync: pickle

**Python serialization** - For preserving complex data structures.

```python
# Note: Pickle reading requires custom implementation
# Use JsonlReader or ParquetReader for standard workflows
from nemo_curator.stages.text.io.reader import JsonlReader

# Convert pickle to JSONL first, then use JsonlReader
reader = JsonlReader(file_paths="converted_data.jsonl")
```

{bdg-secondary}`python-native` {bdg-secondary}`object-preservation`

:::

:::{tab-item} Custom
:sync: custom

**Custom formats** - Extensible framework for specialized file readers.

```python
# Custom file format via custom download/iteration stages
# See custom data loading guide for full implementation
from your_custom_module import CustomDataStage

custom_stage = CustomDataStage(
    file_paths="custom_data.ext",
    custom_params="configuration"
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
from nemo_curator.stages.text.io.writer import JsonlWriter

# Basic export
writer = JsonlWriter(path="output_directory/")

# Add writer to pipeline after processing stages
pipeline.add_stage(writer)

# Execute pipeline to write results
results = pipeline.run(executor)
```

{bdg-secondary}`human-readable` {bdg-secondary}`debugging-friendly`

:::

:::{tab-item} Parquet Export
:sync: parquet-export

**Parquet export** - Optimized columnar format for production workflows.

```python
from nemo_curator.stages.text.io.writer import ParquetWriter

# Basic export
writer = ParquetWriter(path="output_directory/")

# Add writer to pipeline after processing stages
pipeline.add_stage(writer)

# Execute pipeline to write results
results = pipeline.run(executor)
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
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader

# Combine multiple directories in a single reader
reader = JsonlReader(file_paths=[
    "dataset_v1/",
    "dataset_v2/",
    "additional_data/"
])

# For mixed file types, use separate pipelines
# Pipeline 1: JSONL data
jsonl_pipeline = Pipeline(name="jsonl_processing")
jsonl_pipeline.add_stage(JsonlReader(file_paths="text_data.jsonl"))

# Pipeline 2: Parquet data  
parquet_pipeline = Pipeline(name="parquet_processing")
parquet_pipeline.add_stage(ParquetReader(file_paths="structured_data.parquet"))

# Execute pipelines separately, then combine results if needed
```

{bdg-secondary}`data-aggregation` {bdg-secondary}`multi-source`

:::

:::{tab-item} Performance Optimization
:sync: performance

**Performance optimization** - Maximize throughput and reduce memory usage.

```python
# Optimize Parquet reading with PyArrow
reader = ParquetReader(
    file_paths="large_dataset.parquet",
    fields=["text", "id"],  # Column selection for efficiency
    files_per_partition=4,   # Optimize partition size
    read_kwargs={
        "engine": "pyarrow",
        "dtype_backend": "pyarrow"
    }
)

# Optimize memory usage with blocksize
reader = JsonlReader(
    file_paths="data.jsonl",
    blocksize="512MB",  # Adjust based on available memory
    files_per_partition=8
)

# Parallel loading with optimal partitioning
reader = ParquetReader(
    file_paths="data/",
    files_per_partition=16  # Match CPU/GPU count
)
```

{bdg-secondary}`high-performance` {bdg-secondary}`memory-efficient`

:::

:::{tab-item} Large Datasets
:sync: large-datasets

**Working with large datasets** - Handle massive datasets efficiently.

```python
# Efficient processing for large datasets
reader = ParquetReader(
    file_paths="massive_dataset/",
    files_per_partition=8,  # Optimize for cluster size
    blocksize="1GB"        # Large blocks for efficiency
)

# Add to pipeline with processing stages
pipeline = Pipeline(name="large_dataset_processing")
pipeline.add_stage(reader)

# Add memory-efficient processing stages
from nemo_curator.stages.text.modules import ScoreFilter
pipeline.add_stage(ScoreFilter(...))

# Execute with appropriate executor for scale
from nemo_curator.backends.ray.executor import RayExecutor
executor = RayExecutor()
results = pipeline.run(executor)
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

The data acquisition process produces `DocumentBatch` tasks that integrate seamlessly with the pipeline-based processing concepts covered on this page.