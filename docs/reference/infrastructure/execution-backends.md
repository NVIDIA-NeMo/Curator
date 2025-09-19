---
description: "Choose and configure execution backends for NeMo Curator pipelines"
categories: ["reference"]
tags: ["executors", "xenna", "ray", "ray-data", "actor-pool", "pipelines"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

<!-- TODO: further elaborate on what Xenna is and what Ray Data is, and detailed explanations for each parameter -->

(reference-execution-backends)=

# Pipeline Execution Backends

Executors run NeMo Curator `Pipeline` workflows across your compute resources. This reference explains the available backends and how to configure them. It applies to all modalities (text, image, video, and audio).

## How it Works

Build your pipeline by adding stages, then run it with an executor:

```python
from nemo_curator.pipeline import Pipeline

pipeline = Pipeline(name="example_pipeline", description="Curator pipeline")
pipeline.add_stage(...)

# Choose an executor below and run
results = pipeline.run(executor)
```

## Available Backends

### `XennaExecutor` (recommended)

```python
from nemo_curator.backends.xenna import XennaExecutor

executor = XennaExecutor(
    config={
        # 'streaming' (default) or 'batch'
        "execution_mode": "streaming",
        # seconds between status logs
        "logging_interval": 60,
        # continue on failures
        "ignore_failures": False,
        # CPU allocation ratio (0-1)
        "cpu_allocation_percentage": 0.95,
        # streaming autoscale interval (seconds)
        "autoscale_interval_s": 180,
    }
)

results = pipeline.run(executor)
```

- Pass options via `config`; they map to the executor’s pipeline configuration.
- For more details, refer to the official [NVIDIA Cosmos-Xenna project](https://github.com/nvidia-cosmos/cosmos-xenna/tree/main).

### `RayDataExecutor` (experimental)

```python
from nemo_curator.backends.experimental.ray_data import RayDataExecutor

executor = RayDataExecutor()
results = pipeline.run(executor)
```

- Emits an experimental warning; the API and performance characteristics may change.

### `RayActorPoolExecutor` (production-ready)

```python
from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor

executor = RayActorPoolExecutor(
    config={
        "reserved_cpus": 0.0,    # Reserved CPU resources
        "reserved_gpus": 0.0,    # Reserved GPU resources
    }
)
results = pipeline.run(executor)
```

- Used in production for NVIDIA NeMo Curator's deduplication workflows
- Provides better resource management through Ray's ActorPool
- Supports fault tolerance and load balancing
- Recommended for compute-intensive tasks like large-scale classification

## Choosing a Backend

All three options can deliver strong performance; choose based on API fit and maturity:

- **`XennaExecutor`**: default for most workloads due to maturity and extensive real‑world usage (including video pipelines); supports streaming and batch execution with auto‑scaling.
- **`RayDataExecutor` (experimental)**: uses Ray Data API for scalable data processing; the interface is still experimental and may change.
- **`RayActorPoolExecutor`**: production-ready Ray-based executor used in NVIDIA NeMo Curator's own workflows; provides better resource management for compute-intensive tasks.

### Image Curation Recommendations

For image curation workloads, consider these executor-specific optimizations:

**Large-Scale Image Processing (>10K images)**

```python
# Recommended: RayActorPoolExecutor for resource management
from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor

executor = RayActorPoolExecutor(config={
    "reserved_cpus": 1.0,        # Reserve CPUs for coordination
    "reserved_gpus": 0.1,        # Reserve GPU memory for system
})
```

**GPU-Intensive Image Workloads (DALI, embeddings, classification)**

```python
# Use XennaExecutor with streaming for memory efficiency
from nemo_curator.backends.xenna import XennaExecutor

executor = XennaExecutor(config={
    "execution_mode": "streaming",     # Better memory management
    "cpu_allocation_percentage": 0.8,  # Leave headroom for GPU operations
    "autoscale_interval_s": 120,       # Faster scaling for GPU workloads
})
```

**Development and Testing**

```python
# Use RayDataExecutor for rapid prototyping
from nemo_curator.backends.experimental.ray_data import RayDataExecutor

executor = RayDataExecutor()  # Simple setup, good for small datasets
```

## Ray Pipeline Management

All Ray-based executors (`RayDataExecutor` and `RayActorPoolExecutor`) implement automatic resource cleanup and pipeline ending features introduced in recent updates:

### Automatic Resource Cleanup

**Actor Pool Management**

- Actors are automatically cleaned up after each stage completion
- Named actors are properly terminated to prevent resource leaks
- Actor pools are recreated for each stage to ensure clean state

**Pipeline Shutdown**

- `ray.shutdown()` is called automatically in `finally` blocks
- Environment variables are properly reset after pipeline completion
- GPU memory and CUDA contexts are released correctly

### Resource Management Best Practices

**For Long-Running Pipelines**

```python
from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor

# Configure resource reservations to prevent OOM
executor = RayActorPoolExecutor(config={
    "reserved_cpus": 2.0,    # Reserve CPUs for Ray coordination
    "reserved_gpus": 0.2,    # Reserve GPU memory for system processes
})

try:
    results = pipeline.run(executor)
finally:
    # Cleanup is automatic, but you can add custom cleanup here
    pass
```

**Memory Management for Image Workloads**

```python
# Ray executors automatically manage memory, but you can optimize:
import json
import ray

# Initialize Ray with memory limits (optional)
ray.init(
    object_store_memory=8_000_000_000,  # 8GB object store
    _system_config={
        "automatic_object_spilling_enabled": True,
        "object_spilling_config": json.dumps({
            "type": "filesystem",
            "params": {"directory_path": "/tmp/ray_spill"}
        })
    }
)

# Your pipeline will automatically clean up Ray resources
results = pipeline.run(executor)
```

### Pipeline Lifecycle

1. **Initialization**: Ray is initialized with proper environment variables
2. **Stage Execution**: Each stage runs with dedicated actor pools
3. **Inter-Stage Cleanup**: Actor pools are cleaned up between stages
4. **Pipeline Completion**: Final results are collected and returned
5. **Automatic Shutdown**: Ray resources are released and environment is reset

This ensures that image curation pipelines can run reliably in production environments without manual resource management.

## Minimal End-to-End example

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor

# Build your pipeline
pipeline = Pipeline(name="curator_pipeline")
# pipeline.add_stage(stage1)
# pipeline.add_stage(stage2)

# Run with Xenna (recommended)
executor = XennaExecutor(config={"execution_mode": "streaming"})
results = pipeline.run(executor)

print(f"Completed with {len(results) if results else 0} output tasks")
```
