---
description: "Guide to leveraging NVIDIA GPU acceleration in NeMo Curator for faster data processing and memory optimization"
categories: ["reference"]
tags: ["gpu-accelerated", "cuda", "rmm", "performance", "memory-management", "optimization"]
personas: ["mle-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "reference"
modality: "universal"
---

(reference-infra-gpu-processing)=
# GPU Processing Guide

This guide explains how to use GPU acceleration in NVIDIA NeMo Curator for faster text data processing.

## Setting Up GPU Support

To use GPU acceleration, you'll need:
1. NVIDIA GPU with CUDA support
2. RAPIDS libraries installed (cuDF, RMM)
3. GPU-enabled Dask cluster

### Initializing GPU Support

```python
from nemo_curator.utils.distributed_utils import get_client

# Set up GPU-enabled Dask client
client = get_client(cluster_type="gpu")
```

The `get_client` function automatically:
- Creates one worker per available GPU
- Sets up GPU memory management
- Configures RAPIDS memory manager (RMM)

## GPU-Accelerated Modules

NVIDIA NeMo Curator provides these GPU-accelerated modules:

### Deduplication
- Exact deduplication using GPU hashing
- Fuzzy deduplication with GPU-accelerated similarity
- Semantic deduplication using GPU embeddings

### Classification
- Domain classification (English and multilingual)
- Quality classification
- Safety models (AEGIS, Instruction Data Guard)
- Educational content (FineWeb models)
- Content type classification
- Task and complexity classification

## Managing GPU Memory

### Moving Data Between CPU and GPU

Use the `ToBackend` module to move data:
```python
from nemo_curator import Sequential, ToBackend

pipeline = Sequential([
    cpu_operation,          # Runs on CPU
    ToBackend("cudf"),     # Moves to GPU
    gpu_operation,         # Runs on GPU
    ToBackend("pandas"),   # Moves back to CPU
    cpu_operation_2        # Runs on CPU
])
```

### Memory Optimization Tips

1. **Batch Processing**
   - Process data in smaller batches
   - Release GPU memory between operations
   - Monitor GPU memory usage

2. **Strategic Data Movement**
   - Keep data on GPU for multiple GPU operations
   - Move to CPU for memory-intensive operations
   - Use lazy evaluation when possible

3. **Resource Management**
   - Monitor GPU memory with `nvidia-smi`
   - Use RMM memory pools for efficiency
   - Clean up unused GPU objects
