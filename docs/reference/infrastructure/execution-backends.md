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

### `RayActorPoolExecutor` (experimental)

```python
from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor

executor = RayActorPoolExecutor()
results = pipeline.run(executor)
```

## Ray Executors in Practice

Ray-based executors provide enhanced scalability and performance for large-scale data processing tasks. They're beneficial for:

- **Large-scale classification tasks**: Distributed inference across multi-GPU setups
- **Deduplication workflows**: Parallel processing of document similarity computations  
- **Resource-intensive stages**: Automatic scaling based on computational demands

### When to Use Ray Executors

Consider Ray executors when:

- Processing datasets that exceed single-machine capacity
- Running GPU-intensive stages (classifiers, embedding models, etc.)
- Needing automatic fault tolerance and recovery
- Scaling across multi-node clusters

### Ray vs. Xenna Executors

| Feature | XennaExecutor | Ray Executors |
|---------|---------------|---------------|
| **Maturity** | Production-ready | Experimental |
| **Streaming** | Native support | Limited |
| **Resource Management** | Optimized for video/multimodal | General-purpose |
| **Fault Tolerance** | Built-in | Ray-native |
| **Scaling** | Auto-scaling | Manual configuration |

**Recommendation**: Use `XennaExecutor` for production workloads and Ray executors for experimental large-scale processing.

:::{note}
Ray executors emit an experimental warning as the API and performance characteristics may change.
:::

## Choosing a Backend

Both options can deliver strong performance; choose based on API fit and maturity:

- **`XennaExecutor`**: Default for most workloads due to maturity and extensive real-world usage (including video pipelines); supports streaming and batch execution with auto-scaling.
- **Ray Executors (experimental)**: Use Ray Data API for scalable data processing; the interface is still experimental and may change.

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
