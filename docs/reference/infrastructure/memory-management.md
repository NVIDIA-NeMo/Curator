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

This guide explains existing implementations and strategies for managing memory when processing large text datasets with NVIDIA NeMo Curator.

## Memory Challenges in Data Curation

Processing large-scale datasets for LLM training presents unique memory management challenges:

- **Dataset Scale**: Modern LLM training datasets can exceed petabytes, far larger than available RAM/VRAM on any single machine or even cluster. Efficient streaming and batching are essential to process data incrementally.

- **Memory-Intensive Operations**: Tasks like fuzzy deduplication, embedding generation, and classification require loading large models into GPU memory while simultaneously processing document batches, creating competing demands for limited resources.

- **Long-Running Pipelines**: Processing billions of documents can take days or weeks. Even small memory leaks accumulate over time, potentially causing worker crashes or degraded performance. Automatic worker recycling helps mitigate this.

- **Distributed Resource Allocation**: In multi-node clusters, balancing CPU, GPU, and memory resources across workers becomes complex. Different pipeline stages have different resource requirements (such as I/O-heavy readers compared to GPU-heavy classifiers), requiring intelligent allocation.

- **Variable Data Sizes**: Individual documents can range from a few bytes to megabytes. Processing batches of highly variable-sized documents can cause unpredictable memory spikes if not properly managed.

NeMo Curator addresses these challenges through automatic resource management, streaming execution, and configurable batching parameters that you'll learn about in this guide.

## Memory Management in Curator

### Pipeline and Executor Architecture

NeMo Curator uses a **Pipeline** and **Executor** architecture to manage resource allocation and distribute work across compute resources efficiently.

#### How It Works

**1. Pipeline Composition**

The `Pipeline` class provides a high-level abstraction for composing data processing workflows:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

pipeline = Pipeline(
    name="my_pipeline",
    description="Process text documents"
)
pipeline.add_stage(JsonlReader(file_paths="input/"))

# Add text processing stages
# pipeline.add_stage(...)

pipeline.add_stage(JsonlWriter(path="output/"))

# Execute the pipeline
pipeline.run()
```

Each stage declares its resource requirements through the `Resources` class that the executor uses for allocation.

**2. Resource Declaration**

Stages declare their computational needs using the `Resources` dataclass:

```python
from nemo_curator.stages.resources import Resources

# CPU-only stage
cpu_only_resources = Resources(cpus=2.0)
pipeline.add_stage(MyCpuStage(...).with_(resources=cpu_only_resources))

# GPU stage with memory requirement
single_gpu_resources = Resources(
    cpus=4.0,
    gpu_memory_gb=8.0  # GPU memory required in GB (only for single-GPU stages)
)
pipeline.add_stage(MySingleGpuStage(...).with_(resources=single_gpu_resources))

# Multi-GPU stage
multi_gpu_resources = Resources(
    cpus=8.0,
    gpus=2.0  # Request 2 full GPUs
)
pipeline.add_stage(MyMultiGpuStage(...).with_(resources=multi_gpu_resources))
```

Curator automatically allocates memory based on available hardware.

**3. Executor Backends**

Executors handle the actual distribution and execution of work. Curator supports multiple executor backends, with the default being the `XennaExecutor`:

```python
from nemo_curator.backends.xenna import XennaExecutor

executor = XennaExecutor(config={
    "execution_mode": "streaming",  # or "batch"
    "cpu_allocation_percentage": 0.95,  # Reserve 5% for system
    "autoscale_interval_s": 180,  # Adjust workers every 3 minutes
    "logging_interval": 60  # Log status every minute
})

pipeline.run(executor=executor)
```

Refer to the {ref}`Pipeline Execution Backends <reference-execution-backends>` page for more information about Curator's executors.

**4. Worker Management**

Executors automatically manage workers based on stage resource requirements:

- **Worker Allocation**: Creates workers with the exact resources each stage declares
- **Setup/Teardown**: Calls `setup()` once per worker (such as load models) and `teardown()` for cleanup
- **Setup on Node**: Calls `setup_on_node()` once per node (such as download model weights)
- **Task Batching**: Processes multiple tasks per worker call based on `batch_size`
- **Auto-scaling**: Dynamically adjusts worker count based on workload

**5. Memory-Efficient Execution**

The executor ensures memory efficiency through:

- **Lazy Evaluation**: Data flows through the pipeline stage-by-stage without materializing entire datasets
- **Batched Processing**: Stages process data in configurable batch sizes to control memory usage
- **Resource Isolation**: Each worker gets isolated resources preventing interference
- **Automatic Cleanup**: Workers are recycled periodically to prevent memory leaks

## Memory Management Strategies

The previous section discussed how Curator handles resource and worker allocations when executing a pipeline. **In most cases, you don't need to configure `Resources` or executors directly.** Curator automatically:

- Allocates appropriate resources for each stage based on its requirements
- Uses the `XennaExecutor` by default when running pipelines
- Manages worker lifecycle and scaling

**The primary way to control memory usage is by configuring data batch sizes** through reader parameters like `files_per_partition` and `blocksize`. These settings determine how much data flows into each stage at a time, directly impacting memory consumption across your entire pipeline.

Below, we highlight practical ways to configure batch sizes and memory-aware operations.

### 1. Batch Processing

Process data in manageable chunks by controlling file partitioning:

```python
from nemo_curator.stages.text.io.reader import JsonlReader

# Read with controlled partition sizes
reader = JsonlReader(
    file_paths="jsonl_input/",
    files_per_partition=50,  # Process 50 files at a time
    # blocksize="1GB"  # Alternative: control memory usage per data batch
)
```

```python
from nemo_curator.stages.text.io.reader import ParquetReader

# Read with controlled partition sizes
reader = ParquetReader(
    file_paths="parquet_input/",
    files_per_partition=50,  # Process 50 files at a time
    # blocksize="1GB"  # Alternative: control memory usage per data batch
)
```

Setting an appropriate `files_per_partition` or `blocksize` is important because it controls how much data is loaded into memory at once and flows through your pipeline stages. Smaller batches reduce memory usage but may decrease throughput, while larger batches improve processing speed at the cost of higher memory consumption. Choose values based on your available memory and dataset characteristics.

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

**Note on Workflows vs. Pipelines**: Deduplication uses **workflows** that automatically handle I/O (reading and writing) internally, rather than requiring explicit reader and writer stages. The `input_blocksize` parameter controls memory usage in the same way as the `blocksize` parameter in `JsonlReader` and `ParquetReader`. For most other operations, you build **pipelines** by explicitly composing reader → processing stages → writer.

#### Classification

```python
from nemo_curator.stages.text.classifiers import QualityClassifier

# Manage classifier memory
classifier = QualityClassifier(
    model_inference_batch_size=64,  # Smaller batches use less memory (default: 256)
    max_chars=3000  # Limit text length to reduce memory usage (default: 6000)
)
```

**Understanding Batch Sizes**: Curator has two levels of batching that serve different purposes:

- **`batch_size`** (stage-level): Controls how many `DocumentBatch` tasks are processed together by a worker. This affects CPU memory and task scheduling efficiency. Most users don't need to modify this.

- **`model_inference_batch_size`** (model-specific): Controls how many individual documents are passed to the model's forward pass at once. This directly affects GPU memory usage during inference. **This is the primary parameter to adjust** when encountering GPU out-of-memory errors or optimizing GPU utilization.

```{note}
If you encounter a `torch.OutOfMemoryError` during model classification, it is almost always because the `model_inference_batch_size` is too large. Try smaller batch sizes to resolve the error.
```

## Memory Monitoring

### Why Monitor Memory?

Memory monitoring is essential for production data curation pipelines, especially when processing large-scale datasets over extended periods. Without monitoring, you may encounter:

- **Silent Performance Degradation**: Memory leaks can gradually slow down processing without obvious errors
- **Unexpected Failures**: Out-of-memory crashes can occur hours or days into long-running jobs
- **Resource Waste**: Underutilized workers consume resources without contributing to throughput
- **Difficult Debugging**: Without historical data, it's hard to identify which pipeline stage caused a memory issue

Effective monitoring helps you:
- **Detect Issues Early**: Identify memory leaks or inefficient stages before they cause failures
- **Optimize Resource Allocation**: Adjust worker counts and batch sizes based on actual usage patterns
- **Plan Capacity**: Understand resource requirements for scaling to larger datasets
- **Debug Failures**: Investigate what happened leading up to a crash using historical metrics

### Monitoring Stack: Prometheus and Grafana

NeMo Curator supports integration with **Prometheus** and **Grafana**, the industry-standard open-source monitoring stack:

**Prometheus** is a time-series database and monitoring system that:
- Collects metrics from your pipeline at regular intervals (for example, every 15 seconds)
- Stores metrics like CPU usage, GPU memory, worker counts, and task throughput
- Provides a query language (PromQL) to aggregate and analyze metrics
- Runs as a standalone service that "scrapes" metrics exposed by Curator workers

**Grafana** is a visualization platform that:
- Connects to Prometheus as a data source
- Displays metrics in customizable dashboards with graphs, gauges, and alerts
- Provides real-time views of your pipeline's health and performance
- Allows you to set up alerts (for example, notify when GPU memory exceeds 90%)

**How They Work Together**:
1. Curator workers expose metrics in a format Prometheus understands
2. Prometheus periodically scrapes these metrics and stores them
3. Grafana queries Prometheus and displays the data in dashboards
4. You view the dashboards to monitor your pipeline in real-time and historically

### Key Metrics to Monitor

When running production pipelines, track these critical metrics:

- **CPU Memory Usage**: Total RAM consumption across workers to prevent out-of-memory errors
- **GPU Memory Usage**: VRAM consumption per GPU for model-based stages (classifiers, embedders)
- **Worker Count**: Number of active workers per stage to verify proper scaling
- **Task Throughput**: Documents or batches processed per second to measure pipeline efficiency
- **Stage Latency**: Time spent in each pipeline stage to identify bottlenecks
- **Error Rates**: Failed tasks or worker crashes to detect stability issues

### Setting Up Monitoring

Refer to [NeMo Curator Metrics](https://github.com/NVIDIA-NeMo/Curator/tree/main/nemo_curator/metrics) for information about how to use Prometheus and Grafana with NeMo Curator.

## Best Practices

1. **Monitor Memory Usage**
   - **During Development** Use system monitoring tools (`htop`, `nvidia-smi`, `watch -n 1 nvidia-smi`) to observe memory usage patterns as your pipeline runs. Start with small datasets to identify memory bottlenecks before scaling up.
   - **In Production** Set up monitoring dashboards using Prometheus and Grafana (refer to [Memory Monitoring](#memory-monitoring) section above) to track CPU/GPU memory usage, worker utilization, and pipeline throughput over time.
   - **Ray Dashboard** If using Ray-based executors, access the Ray dashboard (typically at `http://localhost:8265`) to view real-time resource usage, task execution, and memory consumption across workers.

2. **Optimize Data Loading**
   - **Split large files into smaller files before curation** If you have individual files that are very large (for example, a single 50 GB JSONL file), you should split them into smaller files (for example, 100 × 500 MB files) before processing. The `blocksize` parameter controls how much data is read into memory at once but does **not** automatically split large files. Pre-splitting ensures better parallelization and prevents memory issues.
   - Control partition sizes via `files_per_partition` or `blocksize` to manage how much data flows through your pipeline

3. **Resource Management**
   - **Use Context Managers**: Always use `with` statements for file operations and resource allocation to ensure proper cleanup even if errors occur.
   - **Clean Up Large Objects**: When working with large datasets in custom stages, explicitly delete temporary objects (e.g., `del large_dataframe`) and consider calling `gc.collect()` after processing large batches to free memory immediately rather than waiting for automatic garbage collection.
   - **GPU Memory**: For GPU-based stages, PyTorch may cache GPU memory. If you encounter GPU out-of-memory errors despite having sufficient GPU capacity, try `torch.cuda.empty_cache()` between stages to clear the cache.
   - **Worker Lifecycle**: Xenna automatically recycles workers periodically (controlled by `worker_max_lifetime_m` and `worker_restart_interval_m` in stage configs) to prevent memory leaks from accumulating during long-running pipelines.
