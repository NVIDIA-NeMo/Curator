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
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.filters.heuristic import WordCountFilter

# Initialize Ray client
ray_client = RayClient()
ray_client.start()

# Create pipeline for processing existing JSONL files
pipeline = Pipeline(name="jsonl_data_processing")

# Read JSONL files
reader = JsonlReader(
    file_paths="/path/to/data",
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

# Add more stages to pipeline...

# Execute pipeline
results = pipeline.run()

# Stop Ray client
ray_client.stop()
```

:::

:::{tab-item} Parquet Reader
:sync: parquet

## Example: Read Parquet and Filter

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.filters.heuristic import WordCountFilter

# Initialize Ray client
ray_client = RayClient()
ray_client.start()

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

# Add more stages to pipeline...

# Execute pipeline
results = pipeline.run()

# Stop Ray client
ray_client.stop()
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
  - Number of files per partition. Overrides `blocksize` if both are provided.
  - None
* - `blocksize`
  - int | str | None
  - Target partition size (e.g., "128MB"). Ignored if `files_per_partition` is provided.
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

- **PyArrow Engine**: Uses `pyarrow` engine by default for better performance.
- **Storage Options**: Supports cloud storage through `storage_options` in `read_kwargs`.
- **Schema Handling**: Automatic schema inference and validation.
- **Columnar Efficiency**: Optimized for reading specific columns.

### Performance Tips

- Use the `fields` parameter to read only the required columns for better performance.
- Set `files_per_partition` based on your cluster size and memory constraints.
- Use the `blocksize` parameter for fine-grained control over partition sizes.

### Memory Tips

:::{warning}
If you set the `blocksize` parameter to a size smaller than your input file size(s), Curator does not split the input files and instead attempts to read each file in full. To avoid out-of-memory issues, use the helper script described below.
:::

If any of your individual JSONL or Parquet files are greater than 2 GiB, we recommend using the `nemo_curator/utils/split_large_files.py` helper script to split them into more manageable sizes and prevent out-of-memory issues. You can run it with:

```bash
python nemo_curator/utils/split_large_files.py --input-path "/path/to/input/dir" --file-type "parquet" --output-path "/path/to/output/dir" --target-size-mb 128
```

It supports splitting JSONL or Parquet files as specified by the `--file-type` argument.

Another option is running file splitting within your existing script. For example, you can split large JSONL files with:

```python
import ray
from nemo_curator.core.client import RayClient
from nemo_curator.utils.split_large_files import split_jsonl_file_by_size

# Start Ray client as usual
ray_client = RayClient()
ray_client.start()

input_files = []  # your list of input jsonl files

ray.get(
  [
    split_jsonl_file_by_size.remote(
      input_file=f,
      output_path="/path/to/output/dir",
      target_size_mb=128,
    )
    for f in input_files
  ]
)

# initialize your Curator pipeline with JsonlReader, etc.
```

Similarly for Parquet files:

```python
import ray
from nemo_curator.core.client import RayClient
from nemo_curator.utils.split_large_files import split_parquet_file_by_size

# Start Ray client as usual
ray_client = RayClient()
ray_client.start()

input_files = []  # your list of input parquet files

ray.get(
  [
    split_parquet_file_by_size.remote(
      input_file=f,
      output_path="/path/to/output/dir",
      target_size_mb=128,
    )
    for f in input_files
  ]
)

# initialize your Curator pipeline with ParquetReader, etc.
```

## Output Integration

Both readers produce `DocumentBatch` tasks that integrate seamlessly with:

- **Processing Stages**: Apply filters, transformations, and quality checks.
- **Writer Stages**: Export to JSONL, Parquet, or other formats.
- **Analysis Tools**: Convert to Pandas/PyArrow for inspection and debugging.
