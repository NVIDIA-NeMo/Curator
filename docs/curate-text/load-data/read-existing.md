---
description: "Read existing JSONL and Parquet datasets using Curator's reader stages."
categories: ["how-to-guides"]
tags: ["jsonl", "parquet", "data-loading", "reader", "pipelines"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "text-only"
---

(text-load-data-read-existing)=

# Read Existing Data

Use Curator's `JsonlReader` and `ParquetReader` to read existing datasets into a pipeline, then optionally add processing stages.

::::{tab-set}

:::{tab-item} JSONL Reader
:sync: jsonl

## Example: Read JSONL and Filter

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import WordCountFilter

# Create pipeline for processing existing JSONL files
pipeline = Pipeline(name="jsonl_data_processing")

# Read JSONL files
reader = JsonlReader(
    file_paths="/path/to/data/*.jsonl",
    files_per_partition=4,
    fields=["text", "url"]  # Only read specific columns
)
pipeline.add_stage(reader)

# Add filtering stage
word_filter = ScoreFilter(
    filter_obj=WordCountFilter(min_words=50, max_words=1000),
    text_field="text"
)
pipeline.add_stage(word_filter)

# Execute pipeline
results = pipeline.run()
```

:::

:::{tab-item} Parquet Reader
:sync: parquet

## Example: Read Parquet and Filter

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import WordCountFilter

# Create pipeline for processing existing Parquet files
pipeline = Pipeline(name="parquet_data_processing")

# Read Parquet files with PyArrow engine
reader = ParquetReader(
    file_paths="/path/to/data",
    files_per_partition=4,
    fields=["text", "metadata"]  # Only read specific columns
)
pipeline.add_stage(reader)

# Add filtering stage
word_filter = ScoreFilter(
    filter_obj=WordCountFilter(min_words=50, max_words=1000),
    text_field="text"
)
pipeline.add_stage(word_filter)

# Execute pipeline
results = pipeline.run()
```

:::

::::

## Reader Configuration

### Common Parameters

Both `JsonlReader` and `ParquetReader` support these configuration options:

```{list-table}
:header-rows: 1
:widths: 20 20 40 20

* - Parameter
  - Type
  - Description
  - Default
* - `file_paths`
  - str | list[str]
  - File paths or glob patterns to read
  - Required
* - `files_per_partition`
  - int | None
  - Number of files per partition
  - None
* - `blocksize`
  - int | str | None
  - Target partition size (e.g., "128MB")
  - None
* - `fields`
  - list[str] | None
  - Column names to read (column selection)
  - None (all columns)
* - `read_kwargs`
  - dict[str, Any] | None
  - Extra arguments for the underlying reader
  - None
```

### Parquet-Specific Features

`ParquetReader` provides these optimizations:

- **PyArrow Engine**: Uses `pyarrow` engine by default for better performance
- **Storage Options**: Supports cloud storage via `storage_options` in `read_kwargs`
- **Schema Handling**: Automatic schema inference and validation
- **Columnar Efficiency**: Optimized for reading specific columns

### Performance Tips

- Use `fields` parameter to read required columns for better performance
- Set `files_per_partition` based on your cluster size and memory constraints
- For cloud storage, configure `storage_options` in `read_kwargs`
- Use `blocksize` for fine-grained control over partition sizes

## Output Integration

Both readers produce `DocumentBatch` tasks that integrate seamlessly with:

- **Processing Stages**: Apply filters, transformations, and quality checks
- **Writer Stages**: Export to JSONL, Parquet, or other formats
- **Analysis Tools**: Convert to Pandas/PyArrow for inspection and debugging
