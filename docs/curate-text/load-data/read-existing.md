---
description: "Read existing JSONL datasets using Curator's reader stage."
categories: ["how-to-guides"]
tags: ["jsonl", "data-loading", "reader", "pipelines"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "text-only"
---

(text-load-data-read-existing)=

# Read Existing Data (JSONL)

Use Curator's `JsonlReader` to read existing JSONL files into a pipeline, then optionally add processing stages.

## Example: Read JSONL and Filter

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.xenna.executor import XennaExecutor
from ray_curator.stages.text.io.reader import JsonlReader
from ray_curator.stages.text.modules import ScoreFilter
from ray_curator.stages.text.filters import WordCountFilter

# Create pipeline for processing existing JSONL files
pipeline = Pipeline(name="custom_data_processing")

# Read JSONL files
reader = JsonlReader(
    file_paths="/path/to/data/*.jsonl",
    files_per_partition=4,
    columns=["text", "url"]  # Only read specific columns
)
pipeline.add_stage(reader)

# Add filtering stage
word_filter = ScoreFilter(
    filter_obj=WordCountFilter(min_words=50, max_words=1000),
    text_field="text"
)
pipeline.add_stage(word_filter)

# Execute pipeline
executor = XennaExecutor()
results = pipeline.run(executor)
```

## Notes

- `JsonlReader` supports `pandas` and `pyarrow` reading backends via the `reader` parameter.
- Use `files_per_partition` or `blocksize` for file partitioning.
- To write outputs, chain a writer like `JsonlWriter` or `ParquetWriter`.
