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

Control how data is split across workers:
```python
from nemo_curator.datasets import DocumentDataset

# Control partition size when reading
dataset = DocumentDataset.read_json(
    files,
    blocksize="256MB",  # Size of each partition
    files_per_partition=10  # Files to group together
)
```

### 2. Batch Processing

Process data in manageable chunks:
```python
from nemo_curator.utils.file_utils import get_batched_files

# Process 50 files at a time
for files in get_batched_files(
    "input/", "output/", "jsonl",
    batch_size=50
):
    batch = DocumentDataset.read_json(files)
    processed = pipeline(batch)
    processed.to_json("output/")
```

### 3. Memory-Aware Operations

Some operations need special memory handling:

#### Deduplication
```python
from nemo_curator.modules import ExactDeduplication

# Control memory usage in deduplication
dedup = ExactDeduplication(
    batch_size=1000,  # Documents per batch
    hash_field="text"
)
```

#### Classification
```python
from nemo_curator.classifiers import QualityClassifier

# Manage classifier memory
classifier = QualityClassifier(
    batch_size=32,  # Smaller batches use less memory
    device="cuda:0"  # Control GPU device
)
```

## Memory Monitoring

### CPU Memory

Monitor system memory:
```python
import psutil

def check_memory():
    mem = psutil.virtual_memory()
    print(f"Memory usage: {mem.percent}%")
    print(f"Available: {mem.available / 1e9:.1f} GB")
```

### GPU Memory

Monitor GPU memory:
```python
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
