---
description: "Strategies and best practices for managing memory when processing large datasets with NeMo Curator"
categories: ["reference"]
tags: ["memory-management", "optimization", "large-scale", "batch-processing", "monitoring", "performance"]
personas: ["mle-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "reference"
modality: "universal"
---

(reference-infra-memory-management)=

# Memory Management Guide

This guide explains strategies for managing memory when processing large text datasets with NVIDIA NeMo Curator.

## Memory Challenges in Text Curation

Processing large text datasets presents several challenges:

- Datasets larger than available RAM/VRAM
- Memory-intensive operations like deduplication
- Long-running processes that may leak memory
- Balancing memory across distributed systems

## Memory Management Strategies

TODO: Talk about how Curator handles resource allocation, etc. in the background.

### 1. Batch Processing

Process data in manageable chunks by controlling file partitioning:

```python
from nemo_curator.stages.text.io.reader import JsonlReader

# Read with controlled partition sizes
reader = JsonlReader(
    file_paths="input/",
    files_per_partition=50,  # Process 50 files at a time
    # blocksize="1GB"  # Alternative: control memory usage per data batch
)
```

```python
from nemo_curator.stages.text.io.reader import ParquetReader

# Read with controlled partition sizes
reader = ParquetReader(
    file_paths="input/",
    files_per_partition=50,  # Process 50 files at a time
    # blocksize="1GB"  # Alternative: control memory usage per data batch
)
```

### 2. Memory-Aware Operations

Some operations need special memory handling:

#### Deduplication

```python
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

# Control memory usage in deduplication
dedup = ExactDeduplicationWorkflow(
    input_path="input/",
    output_path="output/",
    text_field="text",
    input_blocksize="1GB"  # Control memory usage per input block
)
```

#### Classification

```python
from nemo_curator.stages.text.classifiers import QualityClassifier

# Manage classifier memory
classifier = QualityClassifier(
    model_inference_batch_size=64,  # Smaller batches use less memory (default: 256)
    max_chars=3000  # Limit text length to reduce memory usage (default: 6000)
)
```

```{note}
If you encounter a `torch.OutOfMemoryError` during model classification, it is almost always because the `model_inference_batch_size` is too large. Try smaller batch sizes to resolve the error.
```

## Memory Monitoring

TODO: Dashboard

## Best Practices

1. **Monitor Memory Usage**
   - Track memory during development
   - Set up monitoring for production

2. **Optimize Data Loading**
   - Split up large files into smaller chunks before curation (TODO: add section about this)
   - Control partition sizes via `files_per_partition` or `blocksize`

3. **Resource Management**
   - Explicitly release memory after large operations if possible (TODO: consider showing example for this)
   - Use context managers for cleanup
