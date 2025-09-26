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

### 1. Partition Control

Control how data is split across workers using file partitioning:

```python
from nemo_curator.stages.file_partitioning import FilePartitioningStage

# Control partition size when reading
partitioner = FilePartitioningStage(
    file_paths=files,
    blocksize="256MB",  # Target size of each partition in memory
    files_per_partition=10  # Alternative: group files by count instead of size
)
```

### 2. Batch Processing

Process data in manageable chunks by controlling file partitioning:

```python
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

# Read with controlled partition sizes
reader = JsonlReader(
    file_paths="input/",
    files_per_partition=50,  # Process 50 files at a time
    blocksize="1GB"  # Alternative: control memory usage per partition
)

# Process and write in batches
writer = JsonlWriter(path="output/")
```

### 3. Memory-Aware Operations

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

## Memory Monitoring

### CPU Memory

Monitor system memory:

```python
# Note: Requires installing psutil: pip install psutil
import psutil

def check_memory():
    mem = psutil.virtual_memory()
    print(f"Memory usage: {mem.percent}%")
    print(f"Available: {mem.available / 1e9:.1f} GB")
```

### GPU Memory

Monitor GPU memory:

```python
# Note: Requires CUDA installation with nemo_curator[cuda12]
import pynvml

def check_gpu_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory used: {info.used / 1e9:.1f} GB")
```

## Best Practices

1. **Monitor Memory Usage**
   - Track memory during development
   - Set up monitoring for production
   - Handle out-of-memory gracefully

2. **Optimize Data Loading**
   - Use lazy loading when possible
   - Control partition sizes
   - Clean up unused data

3. **Resource Management**
   - Release memory after large operations
   - Use context managers for cleanup
   - Monitor long-running processes
