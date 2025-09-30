---
description: "Implementation guide for resumable processing to handle interrupted large-scale data operations in NeMo Curator"
categories: ["reference"]
tags: ["batch-processing", "large-scale", "optimization", "python-api", "configuration", "monitoring"]
personas: ["mle-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "reference"
modality: "universal"
---

(reference-infra-resumable-processing)=

# Resumable Processing

This guide explains strategies to make large-scale data operations resumable.

## Why Resumable Processing Matters

Large datasets can trigger interruptions due to:

- System timeouts
- Hardware failures
- Network issues
- Resource constraints
- Scheduled maintenance

NeMo Curator provides built-in functionality for resuming operations from where they left off.

## How it Works

The resumption approach works by:

1. Examining filenames in the input directory using `get_all_file_paths_under()`
2. Comparing them with filenames in the output directory
3. Identifying unprocessed files by comparing file counts or specific file lists
4. Rerunning the pipeline on remaining files

This approach works best when you:

- Use consistent directory structures for input and output
- Process files in batches using `files_per_partition` to manage memory usage
- Create checkpoints by writing intermediate results to disk

## Practical Patterns for Resumable Processing

### 1. Process remaining files using directory comparison

Use file listing utilities to identify unprocessed files and process them directly:

```python
from nemo_curator.utils.file_utils import get_all_file_paths_under
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

# Get all input files
input_files = get_all_file_paths_under(
    "input_directory/", 
    recurse_subdirectories=True, 
    keep_extensions=[".jsonl"]
)

# Get already processed output files
output_files = get_all_file_paths_under(
    "output_directory/", 
    recurse_subdirectories=True, 
    keep_extensions=[".jsonl"]
)

# Simple approach: if output directory has fewer files than input, 
# process all remaining inputs
if len(output_files) < len(input_files):
    # Process remaining files
    pipeline = Pipeline(name="resumable_processing")
    
    # Read input files
    reader = JsonlReader(file_paths=input_files, fields=["text", "id"])
    pipeline.add_stage(reader)
    
    # Add your processing stages here
    # pipeline.add_stage(your_processing_stage)
    
    # Write results
    writer = JsonlWriter(path="output_directory/")
    pipeline.add_stage(writer)
    
    # Execute pipeline
    pipeline.run()
```

### 2. Batch processing with file partitioning

Control memory usage and enable checkpoint creation by using NeMo Curator's built-in file partitioning:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

# Process files in smaller batches using files_per_partition
pipeline = Pipeline(name="batch_processing")

# JsonlReader automatically handles file partitioning
reader = JsonlReader(
    file_paths="input_directory/", 
    files_per_partition=64,  # Process 64 files at a time
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Add your processing stages here
# pipeline.add_stage(your_processing_stage)

# Write results
writer = JsonlWriter(path="output_directory/")
pipeline.add_stage(writer)

# Execute pipeline - processes files in batches automatically
pipeline.run()
```
