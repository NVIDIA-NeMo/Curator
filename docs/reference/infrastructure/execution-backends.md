---
description: "Choose and configure execution backends for NeMo Curator pipelines"
categories: ["reference"]
tags: ["executors", "xenna", "ray", "ray-data", "actor-pool", "pipelines"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(reference-execution-backends)=

# Pipeline Execution Backends

Configure and optimize execution backends to run NeMo Curator pipelines efficiently across single machines, multi-GPU systems, and distributed clusters.

## Overview

Execution backends (executors) are the engines that run NeMo Curator `Pipeline` workflows across your compute resources. They handle:

- **Task Distribution**: Distribute pipeline stages across available workers and GPUs
- **Resource Management**: Allocate CPU, GPU, and memory resources to processing tasks
- **Scaling**: Automatically or manually scale processing based on workload
- **Data Movement**: Optimize data transfer between pipeline stages

**Choosing the right executor** impacts:
- Pipeline performance and throughput
- Resource utilization efficiency
- Ease of deployment and monitoring

This guide covers all execution backends available in NeMo Curator and applies to all modalities: text, image, video, and audio curation.

## Basic Usage Pattern

All pipelines follow this standard execution pattern:

```python
from nemo_curator.pipeline import Pipeline

pipeline = Pipeline(name="example_pipeline", description="Curator pipeline")
pipeline.add_stage(...)

# Choose an executor below and run
results = pipeline.run(executor)
```

**Key points:**
- The same pipeline definition works with any executor
- Executor choice is independent of pipeline stages
- Switch executors without changing pipeline code

## Available Backends

### `XennaExecutor` (recommended)

`XennaExecutor` uses Cosmos-Xenna, a Ray-based execution engine optimized for distributed data processing. Xenna provides native streaming support, automatic resource scaling, and built-in fault tolerance. This executor is the recommended choice for most workloads, especially for video and multimodal pipelines.

**Key Features**:
- **Streaming execution**: Process data incrementally as it arrives, reducing memory requirements
- **Auto-scaling**: Dynamically adjusts worker allocation based on stage throughput
- **Fault tolerance**: Built-in error handling and recovery mechanisms
- **Resource optimization**: Efficient CPU and GPU allocation for video/multimodal workloads

```python
from nemo_curator.backends.xenna import XennaExecutor

executor = XennaExecutor(
    config={
        # Execution mode: 'streaming' (default) or 'batch'
        # Batch processes all data for a stage before moving to the next; streaming runs stages concurrently.
        "execution_mode": "streaming",
        
        # Logging interval: seconds between status logs (default: 60)
        # Controls how frequently progress updates are printed
        "logging_interval": 60,
        
        # Ignore failures: whether to continue on failures (default: False)
        # When True, the pipeline continues execution instead of failing fast when stages raise errors.
        "ignore_failures": False,
        
        # CPU allocation percentage: ratio of CPU to allocate (0-1, default: 0.95)
        # Fraction of available CPU resources to use for pipeline execution
        "cpu_allocation_percentage": 0.95,
        
        # Autoscale interval: seconds between auto-scaling checks (default: 180)
        # How often to run the stage auto-scaler.
        "autoscale_interval_s": 180,
        
        # Max workers per stage: maximum number of workers (optional)
        # Limits worker count per stage; None means no limit
        "max_workers_per_stage": None,
    }
)

results = pipeline.run(executor)
```

**Configuration Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `execution_mode` | `str` | `"streaming"` | Execution mode: `"streaming"` for incremental processing or `"batch"` for full dataset processing |
| `logging_interval` | `int` | `60` | Seconds between status log updates |
| `ignore_failures` | `bool` | `False` | If `True`, continue pipeline execution even when stages fail |
| `cpu_allocation_percentage` | `float` | `0.95` | Fraction (0-1) of available CPU resources to allocate |
| `autoscale_interval_s` | `int` | `180` | Seconds between auto-scaling evaluations |
| `max_workers_per_stage` | `int \| None` | `None` | Maximum workers per stage; `None` means no limit |

For more details, refer to the official [NVIDIA Cosmos-Xenna project](https://github.com/nvidia-cosmos/cosmos-xenna/tree/main).

### `RayActorPoolExecutor`

`RayActorPoolExecutor` uses Ray's ActorPool for efficient distributed processing with fine-grained resource management. This executor creates pools of Ray actors per stage, enabling better load balancing and fault tolerance through Ray's native mechanisms. This executor is the recommended choice for GPU-intensive deduplication workflows and stages requiring precise resource allocation.

**Key Features**:
- **ActorPool-based execution**: Creates dedicated actor pools per stage for optimal resource utilization
- **Load balancing**: Uses `map_unordered` for efficient work distribution across actors
- **RAFT support**: Native integration with [RAFT](https://github.com/rapidsai/raft) (RAPIDS Analytics Framework Toolbox) for GPU-accelerated clustering and nearest-neighbor operations
- **Head node exclusion**: Optional `ignore_head_node` parameter to reserve the Ray cluster's [head node](https://docs.ray.io/en/latest/cluster/key-concepts.html#head-node) for coordination tasks only

```python
from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor

executor = RayActorPoolExecutor(
    config={
        # Reserved CPUs: CPUs to reserve on each node (default: 0.0)
        "reserved_cpus": 0.0,
        
        # Reserved GPUs: GPUs to reserve on each node (default: 0.0)
        "reserved_gpus": 0.0,
        
        # Runtime environment: Ray runtime environment configuration
        "runtime_env": {},
    },
    # Exclude head node from task execution (default: False)
    ignore_head_node=False,
)

results = pipeline.run(executor)
```

**Configuration Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reserved_cpus` | `float` | `0.0` | CPUs to reserve on each node for system processes |
| `reserved_gpus` | `float` | `0.0` | GPUs to reserve on each node |
| `runtime_env` | `dict` | `{}` | Ray runtime environment configuration |
| `ignore_head_node` | `bool` | `False` | If `True`, exclude the head node from task execution |

**When to Use**:
- GPU-accelerated deduplication (exact, fuzzy, and semantic)
- Stages requiring RAFT-based clustering (K-means, connected components)
- Workflows needing fine-grained control over actor lifecycle
- Multi-GPU pipelines with heterogeneous resource requirements

:::{dropdown} Example: Fuzzy Deduplication with RayActorPoolExecutor
:icon: code-square

```python
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

workflow = FuzzyDeduplicationWorkflow(
    input_path="/data/documents",
    cache_path="/data/cache",
    output_path="/data/output",
    text_field="text",
    perform_removal=True,
    num_bands=20,
    minhashes_per_band=13,
)

# The workflow automatically uses RayActorPoolExecutor for GPU-accelerated stages
results = workflow.run()
```

For more details, refer to {ref}`Text Deduplication <text-process-data-dedup>`.
:::

### `RayDataExecutor` (Experimental)

`RayDataExecutor` uses Ray Data, a scalable data processing library built on Ray Core. Ray Data provides a familiar DataFrame-like API for distributed data transformations. This executor is experimental and best suited for large-scale batch processing tasks that benefit from Ray Data's optimized data loading and transformation pipelines.

**Key Features**:
- **Ray Data API**: Leverages Ray Data's optimized data processing primitives
- **Scalable transformations**: Efficient map-batch operations across distributed workers
- **Experimental status**: API and performance characteristics may change

```python
from nemo_curator.backends.experimental.ray_data import RayDataExecutor

executor = RayDataExecutor()
results = pipeline.run(executor)
```

:::{note}
`RayDataExecutor` currently has limited configuration options. For more control over execution, consider using `XennaExecutor` or `RayActorPoolExecutor`.
:::

## Ray Executors in Practice

Ray-based executors provide enhanced scalability and performance for large-scale data processing tasks. These executors are beneficial for:

- **Large-scale classification tasks**: Distributed inference across multi-GPU setups
- **Deduplication workflows**: Parallel processing of document similarity computations  
- **Resource-intensive stages**: Automatic scaling based on computational demands

### When to Use Each Executor

| Use Case | Recommended Executor |
|----------|---------------------|
| Streaming video/multimodal pipelines | `XennaExecutor` |
| GPU-accelerated deduplication | `RayActorPoolExecutor` |
| Batch data transformations | `RayDataExecutor` (experimental) |
| General production workloads | `XennaExecutor` |
| RAFT-based clustering operations | `RayActorPoolExecutor` |

### Executor Comparison

| Feature | XennaExecutor | RayActorPoolExecutor | RayDataExecutor |
|---------|---------------|----------------------|-----------------|
| **Maturity** | Stable | Stable | Experimental |
| **Streaming** | Native support | Batch-oriented | Limited |
| **Resource Management** | Optimized for video/multimodal | Fine-grained actor control | General-purpose |
| **Fault Tolerance** | Built-in | Ray-native actor restart | Ray-native |
| **Scaling** | Auto-scaling | Manual configuration | Manual configuration |
| **RAFT Support** | No | Yes | No |
| **Head Node Exclusion** | Not supported | Yes | Yes |

**Recommendation**: Use `XennaExecutor` for streaming and video workloads. Use `RayActorPoolExecutor` for deduplication and RAFT-based operations.

:::{note}
`RayDataExecutor` is experimental and its API may change.
:::

## Choosing a Backend

All three executors can deliver strong performance; choose based on your workload requirements:

- **`XennaExecutor`**: Default for most workloads due to maturity and extensive real-world usage (including video pipelines); supports streaming and batch execution with auto-scaling.
- **`RayActorPoolExecutor`**: Recommended for GPU-accelerated deduplication workflows and RAFT-based operations; provides fine-grained actor lifecycle control.
- **`RayDataExecutor`** (experimental): Best for batch data transformations using Ray Data's DataFrame-like API; the interface may change.

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
