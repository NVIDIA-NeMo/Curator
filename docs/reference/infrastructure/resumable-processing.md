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

This guide explains how to implement resumable processing for large-scale data operations that may be interrupted.

## Why Resumable Processing Matters

When processing large datasets, operations can be interrupted due to:
- System timeouts
- Hardware failures
- Network issues
- Resource constraints
- Scheduled maintenance

NeMo Curator provides built-in functionality for resuming operations from where they left off.

## Key Utilities for Resumable Processing

### 1. `get_remaining_files`

This function identifies files that haven't been processed yet:

```python
from nemo_curator.utils.file_utils import get_remaining_files

# Get only files that haven't been processed yet
files = get_remaining_files("input_directory/", "output_directory/", "jsonl")
dataset = DocumentDataset.read_json(files, add_filename=True)

# Continue processing with unprocessed files only
processed_dataset = my_processor(dataset)
processed_dataset.to_json("output_directory/", write_to_filename=True)
```

### 2. `get_batched_files`

This function returns an iterator that yields batches of unprocessed files:

```python
from nemo_curator.utils.file_utils import get_batched_files

# Process files in batches of 64
for file_batch in get_batched_files("input_directory/", "output_directory/", "jsonl", batch_size=64):
    dataset = DocumentDataset.read_json(file_batch, add_filename=True)
    
    # Process batch
    processed_batch = my_processor(dataset)
    
    # Write results for this batch
    processed_batch.to_json("output_directory/", write_to_filename=True)
```

## How Resumable Processing Works

The resumption system works by:

1. Examining filenames in the input directory
2. Comparing them with filenames in the output directory
3. Identifying files that exist in the input but not in the output directory
4. Processing only those unprocessed files

This approach requires:
- Using `add_filename=True` when reading files
- Using `write_to_filename=True` when writing files
- Maintaining consistent filename patterns between input and output

## Best Practices for Resumable Processing

1. **Preserve filenames**: Use `add_filename=True` when reading files and `write_to_filename=True` when writing.

2. **Batch appropriately**: Choose batch sizes that balance memory usage and processing efficiency.

3. **Use checkpointing**: For complex pipelines, consider writing intermediate results to disk.

4. **Test resumability**: Verify that your process can resume correctly after simulated interruptions.

5. **Monitor disk space**: Ensure sufficient storage for both input and output files.

6. **Log progress**: Maintain logs of processed files to help diagnose issues.
