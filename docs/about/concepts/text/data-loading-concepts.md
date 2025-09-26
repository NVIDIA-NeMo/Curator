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

## Pipeline-Based Data Loading

NeMo Curator uses a **pipeline-based architecture** for handling large-scale text data processing. Data flows through processing stages that transform tasks, enabling distributed processing of local files.

The system provides two primary readers for text data:

- **JsonlReader** - For JSON Lines format files (most common)
- **ParquetReader** - For columnar Parquet files (better performance for large datasets)

Both readers support optimization through:

- **Field selection** - Reading specified columns to reduce memory usage
- **Partitioning control** - Using `block_size` or `files_per_partition` to optimize distributed processing
- **Recommended block size** - Use ~128MB for optimal object store performance with smaller data chunks

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader

# Basic usage with optimization
pipeline = Pipeline(name="data_processing")

# JSONL reader with field selection and partitioning
jsonl_reader = JsonlReader(
    file_paths="/path/to/jsonl_directory",
    blocksize="128MB",  # Recommended for object store optimization
    fields=["text", "id"]  # Column selection for efficiency
)
pipeline.add_stage(jsonl_reader)

# Parquet reader with performance optimization
parquet_reader = ParquetReader(
    file_paths="data.parquet",
    files_per_partition=4,  # Alternative to blocksize
    fields=["text", "metadata"],
    read_kwargs={
        "engine": "pyarrow",
        "dtype_backend": "pyarrow"
    }
)

# Execute pipeline
results = pipeline.run()
```

## Supported File Formats

::::{tab-set}

:::{tab-item} JSONL
:sync: `jsonl`

**JSON Lines format** - Most commonly used format for text datasets.

```python
from nemo_curator.stages.text.io.reader import JsonlReader

# Optimized JSONL reading
reader = JsonlReader(
    file_paths="data_directory/",
    blocksize="128MB",  # Optimal for distributed processing
    fields=["text", "id"]  # Read only required columns
)
```

{bdg-secondary}`most-common` {bdg-secondary}`fast-loading`

:::

:::{tab-item} Parquet
:sync: parquet

**Columnar format** - Better performance for large datasets with PyArrow optimization.

```python
from nemo_curator.stages.text.io.reader import ParquetReader

# Optimized Parquet reading
reader = ParquetReader(
    file_paths="data.parquet",
    files_per_partition=8,  # Control partition size
    fields=["text", "metadata"]  # Column selection
)
```

{bdg-secondary}`production` {bdg-secondary}`gpu-optimized`

:::

::::

## Optimization Strategies

### Partitioning Control

```{note}
**Partitioning Strategy**: Specify either `files_per_partition` or `blocksize`. If `files_per_partition` is provided, `blocksize` is ignored.
```

```python
# Option 1: Size-based partitioning (recommended)
reader = JsonlReader(
    file_paths="/path/to/data",
    blocksize="128MB"  # Optimal for object store performance
)

# Option 2: File count-based partitioning  
reader = ParquetReader(
    file_paths="/path/to/data",
    files_per_partition=16  # Match your cluster size
)
```

### Performance Recommendations

- **Block size**: Use ~128MB for optimal object store performance - smaller objects improve distributed data exchange
- **Field selection**: Specify `fields` parameter to read required columns
- **Engine choice**: ParquetReader defaults to PyArrow with `dtype_backend="pyarrow"` for GPU compatibility
- **Partition sizing**: Match `files_per_partition` to your available CPU/GPU count

## Data Export Options

NeMo Curator provides flexible export options for processed datasets:

```python
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter

# Add writers to pipeline after processing stages
pipeline.add_stage(JsonlWriter(path="output_directory/"))
# or
pipeline.add_stage(ParquetWriter(path="output_directory/"))

# Execute pipeline to write results
results = pipeline.run()
```

```{note}
**Deterministic File Naming**

Writers automatically generate deterministic filenames based on source files processed by readers, ensuring consistent and reproducible output across pipeline runs.
```

## Common Loading Patterns

### Multi-Source Data

```python
# Combine multiple directories
reader = JsonlReader(file_paths=[
    "dataset_v1/",
    "dataset_v2/", 
    "additional_data/"
])

# For different file types, use separate readers in the same pipeline
pipeline = Pipeline(name="multi_format_processing")
pipeline.add_stage(JsonlReader(file_paths="text_data.jsonl"))
pipeline.add_stage(ParquetReader(file_paths="structured_data.parquet"))
```

### Large-Scale Processing

```python
# Efficient processing for massive datasets
reader = ParquetReader(
    file_paths="massive_dataset/",
    blocksize="128MB",  # Optimal chunk size for distributed processing
    fields=["text", "id"]  # Memory efficiency through column selection
)

pipeline = Pipeline(name="large_dataset_processing")
pipeline.add_stage(reader)

# Add processing stages
from nemo_curator.stages.text.modules import ScoreFilter
pipeline.add_stage(ScoreFilter(...))

results = pipeline.run()
```

## Remote Data Acquisition

For users who need to download and process data from remote sources, NeMo Curator provides a comprehensive data acquisition framework. {ref}`Data Acquisition Concepts <about-concepts-text-data-acquisition>` covers this in detail, including:

- **DocumentDownloader, DocumentIterator, DocumentExtractor** components
- **Built-in support** for Common Crawl, ArXiv, Wikipedia, and custom sources  
- **Integration patterns** with DocumentDataset
- **Configuration and scaling** strategies

The data acquisition process produces `DocumentBatch` tasks that integrate seamlessly with the pipeline-based processing concepts on this page.
