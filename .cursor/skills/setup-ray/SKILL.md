---
name: setup-ray
description: |
  Configure Ray clusters for distributed NeMo Curator execution.
  Use when the user wants to scale to multiple nodes, configure
  Ray resources, set up distributed deduplication, or optimize cluster settings.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
disable-model-invocation: true
---

# Ray Cluster Setup

Configure Ray for distributed NeMo Curator execution.

## When to Use

- Scaling beyond single machine
- Multi-node GPU clusters
- Large-scale deduplication (>100GB)
- Production deployments

## Quick Start

### Single Node (Development)

For single-node development, use `RayClient` which manages the cluster lifecycle:

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline

with RayClient(num_cpus=8, num_gpus=2) as client:
    pipeline = Pipeline(name="dev_pipeline", stages=[...])
    results = pipeline.run()
```

**Note**: Do NOT call `ray.init()` directly when using `XennaExecutor` - it handles Ray initialization internally.

### Multi-Node Cluster

For multi-node clusters, start Ray externally and connect via environment variable:

```bash
# On head node
ray start --head --port=6379

# On worker nodes
ray start --address='<head-ip>:6379'

# Set environment variable before running NeMo Curator
export RAY_ADDRESS="<head-ip>:6379"
```

Then run your pipeline - `RayClient` will detect the existing cluster:

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline

# RayClient detects RAY_ADDRESS and skips cluster setup
with RayClient() as client:
    pipeline = Pipeline(name="distributed_pipeline", stages=[...])
    results = pipeline.run()
```

## Ray Configuration in NeMo Curator

NeMo Curator provides `RayClient` to manage Ray cluster lifecycle. The `XennaExecutor` handles `ray.init()` internally, so you typically only need `RayClient` for explicit cluster management.

### Via Pipeline YAML (Recommended)

```yaml
ray_client:
  _target_: nemo_curator.core.client.RayClient
  num_cpus: null  # null = use all available
  num_gpus: 4
  object_store_memory: null  # bytes, null = auto
  ray_dashboard_port: 8265
  ray_port: 6379
  enable_object_spilling: false
```

Run with:

```bash
python -m nemo_curator.config.run --config-path=. --config-name=my_pipeline
```

### Via Python (Context Manager)

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline

# RayClient as context manager (recommended)
with RayClient(num_cpus=None, num_gpus=4) as client:
    pipeline = Pipeline(name="my_pipeline", stages=[...])
    results = pipeline.run()

# Or explicit start/stop
client = RayClient(num_cpus=None, num_gpus=4)
client.start()  # Required before running pipelines
try:
    pipeline = Pipeline(name="my_pipeline", stages=[...])
    results = pipeline.run()
finally:
    client.stop()
```

### RayClient Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_cpus` | int \| None | None | CPUs to use (None = all) |
| `num_gpus` | int \| None | None | GPUs to use (None = all) |
| `ray_port` | int | 6379 | Ray GCS port |
| `ray_dashboard_port` | int | 8265 | Dashboard port |
| `object_store_memory` | int \| None | None | Object store size in bytes |
| `enable_object_spilling` | bool | False | Enable disk spilling |
| `ray_temp_dir` | str | /tmp/ray | Ray temp directory |

## Cluster Sizing Guidelines

### By Dataset Size

| Dataset | Nodes | GPUs/Node | RAM/Node | Notes |
|---------|-------|-----------|----------|-------|
| < 10GB | 1 | 1 | 32GB | Development |
| 10-100GB | 1 | 4 | 64GB | Single node |
| 100-500GB | 2-4 | 4-8 | 128GB | Small cluster |
| 500GB-1TB | 4-8 | 8 | 256GB | Medium cluster |
| > 1TB | 8+ | 8 | 512GB | Large cluster |

### By Workload

| Workload | CPU Requirement | GPU Requirement |
|----------|-----------------|-----------------|
| Heuristic filtering | High | None |
| ML classification | Medium | High |
| Fuzzy deduplication | High | Medium |
| Video processing | Medium | High |
| Captioning | Low | Very High |

## Resource Configuration

### CPU Allocation

```yaml
ray_client:
  _target_: nemo_curator.core.client.RayClient
  num_cpus: 64  # Total CPUs for cluster
```

For Ray Data (XennaExecutor):

```python
executor = XennaExecutor(config={
    "cpu_allocation_percentage": 0.95,  # Use 95% of CPUs
})
```

### GPU Allocation

NeMo Curator stages declare GPU requirements via the `Resources` class:

```python
from nemo_curator.stages.resources import Resources

# Stage with shared GPU (memory fraction)
Resources(gpu_memory_gb=16.0)  # Uses portion of GPU based on available memory

# Stage with dedicated GPU
Resources(gpus=1.0)  # Reserves entire GPU

# Stage with entire GPU plus hardware encoders/decoders (video processing)
Resources(entire_gpu=True)  # Gets full GPU including NVDEC/NVENC

# Stage requiring hardware video decoder
Resources(nvdecs=1)  # Allocates 1 NVDEC unit

# Stage requiring hardware video encoder
Resources(nvencs=1)  # Allocates 1 NVENC unit
```

**Resources Attributes**:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `cpus` | float | 1.0 | CPU cores required |
| `gpu_memory_gb` | float | 0.0 | GPU memory (single-GPU stages) |
| `gpus` | float | 0.0 | GPU count (multi-GPU stages) |
| `entire_gpu` | bool | False | Allocate full GPU with NVDEC/NVENC |
| `nvdecs` | int | 0 | NVDEC hardware decoder units |
| `nvencs` | int | 0 | NVENC hardware encoder units |

**Note**: Cannot specify both `gpus` and `gpu_memory_gb` simultaneously.

### Memory Configuration

Configure memory via `RayClient` (recommended) or Ray CLI for external clusters:

```python
from nemo_curator.core.client import RayClient

# Via RayClient
with RayClient(
    object_store_memory=100 * 1024**3,  # 100GB object store
    enable_object_spilling=True,  # Spill to disk when full
) as client:
    # Run pipeline...
```

For external clusters started via CLI:

```bash
ray start --head \
    --object-store-memory=107374182400 \
    --system-config='{"object_spilling_threshold": 0.8}'
```

## Executor Configuration

### XennaExecutor (Default)

`XennaExecutor` is the default executor that runs pipelines using Cosmos-Xenna. It handles `ray.init()` internally, so you should NOT call `ray.init()` directly when using it.

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline

# RayClient manages cluster, XennaExecutor runs the pipeline
with RayClient(num_gpus=4) as client:
    executor = XennaExecutor(config={
        "logging_interval": 60,           # Log every 60 seconds
        "ignore_failures": False,         # Stop on errors
        "execution_mode": "streaming",    # For large datasets
        "cpu_allocation_percentage": 0.95,
        "autoscale_interval_s": 180,      # Autoscaling interval
    })
    
    pipeline = Pipeline(name="my_pipeline", stages=[...])
    results = pipeline.run(executor=executor)
```

### Execution Modes

| Mode | Memory | Throughput | Use Case |
|------|--------|------------|----------|
| `batch` | Higher | Higher | Small datasets, reprocessing |
| `streaming` | Lower | Lower | Large datasets, memory-limited |

## Multi-Node Setup

### Head Node

```bash
# Start head node
ray start --head \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --num-cpus=64 \
  --num-gpus=8

# Get connection info
ray status
```

### Worker Nodes

```bash
# Connect to head node
ray start --address='<head-ip>:6379' \
  --num-cpus=64 \
  --num-gpus=8
```

### Verify Cluster

Use this snippet to verify your cluster is running correctly (for diagnostics only, not for running pipelines):

```python
import ray
ray.init(address='auto')  # Connect to existing cluster

print(f"Nodes: {len(ray.nodes())}")
print(f"Total CPUs: {ray.cluster_resources().get('CPU', 0)}")
print(f"Total GPUs: {ray.cluster_resources().get('GPU', 0)}")

ray.shutdown()  # Disconnect after verification
```

## Slurm Integration

For HPC environments with Slurm:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

# Start Ray on head node
if [[ $SLURM_PROCID -eq 0 ]]; then
    ray start --head --port=6379
    sleep 10
fi

# Start workers
if [[ $SLURM_PROCID -ne 0 ]]; then
    ray start --address=$HEAD_NODE:6379
fi

# Wait for cluster
sleep 30

# Run NeMo Curator
python -m nemo_curator.config.run --config-path=. --config-name=pipeline
```

## Performance Tuning

### Shuffle Optimization

For deduplication (shuffle-heavy workloads), tune `bands_per_iteration` to control memory usage:

```python
from nemo_curator.stages.deduplication.fuzzy import FuzzyDeduplicationWorkflow

# Lower bands_per_iteration reduces memory but increases iterations
workflow = FuzzyDeduplicationWorkflow(
    bands_per_iteration=1,  # Minimum memory usage (default: 5)
    # ... other params
)
```

### Network Configuration

For multi-node clusters:

```bash
# Use high-speed interconnect
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_OBJECT_MANAGER_PULL_TIMEOUT_MS=120000

ray start --head \
  --object-manager-port=8076 \
  --node-manager-port=8077
```

### Object Store Tuning

For memory-intensive stages, configure via `RayClient` or Ray CLI:

```python
from nemo_curator.core.client import RayClient

# Configure object store via RayClient
with RayClient(
    object_store_memory=100 * 1024**3,  # 100GB
    enable_object_spilling=True,
) as client:
    # Run pipeline...
```

Or via Ray CLI for external clusters:

```bash
ray start --head \
    --object-store-memory=107374182400 \
    --plasma-directory=/dev/shm
```

## Monitoring

### Ray Dashboard

```bash
# Start with dashboard
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

# Access at http://<head-ip>:8265
```

Dashboard shows:
- Node status and resources
- Job status and logs
- Memory usage
- Task timeline

### Metrics to Watch

| Metric | Healthy Range | Action if Unhealthy |
|--------|---------------|---------------------|
| CPU utilization | >80% | Normal |
| GPU utilization | >80% for GPU stages | Increase batch size |
| Memory usage | <90% | Reduce batch size, add nodes |
| Object store | <80% | Reduce parallelism |

### Command-Line Monitoring

```bash
# Cluster status
ray status

# Node info
ray nodes

# Resource usage
watch -n 5 "ray status | grep -A5 'Resources'"
```

## Troubleshooting

### Connection Issues

```bash
# Check Ray is running
ray status

# Check ports
netstat -tlnp | grep 6379

# Restart cluster
ray stop
ray start --head
```

### Out of Memory

1. Reduce `bands_per_iteration` for deduplication
2. Use `execution_mode: streaming`
3. Add more worker nodes
4. Increase object store memory

### Slow Performance

1. Check network bandwidth between nodes
2. Verify GPU utilization with `nvidia-smi`
3. Check for stragglers in Ray dashboard
4. Increase parallelism if CPU-bound

### Task Failures

Enable detailed logging via environment variables before starting Ray:

```bash
export RAY_BACKEND_LOG_LEVEL=debug

# Then start RayClient or ray CLI
```

Or check Ray dashboard logs at `http://<head-ip>:8265` for task-level debugging.

## Example Configurations

### Development (Single Node)

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline

with RayClient(num_cpus=8, num_gpus=1) as client:
    executor = XennaExecutor(config={
        "execution_mode": "batch",
    })
    
    pipeline = Pipeline(name="dev_pipeline", stages=[...])
    results = pipeline.run(executor=executor)
```

### Production (Multi-Node)

```bash
# Start external cluster first
ray start --head --port=6379
export RAY_ADDRESS="<head-ip>:6379"
```

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline

# RayClient detects existing cluster via RAY_ADDRESS
with RayClient() as client:
    executor = XennaExecutor(config={
        "execution_mode": "streaming",
        "cpu_allocation_percentage": 0.95,
        "autoscale_interval_s": 180,
        "ignore_failures": False,
    })
    
    pipeline = Pipeline(name="prod_pipeline", stages=[...])
    results = pipeline.run(executor=executor)
```

### Large-Scale Deduplication

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy import FuzzyDeduplicationWorkflow

# Configure for large memory workloads
with RayClient(
    object_store_memory=100 * 1024**3,
    enable_object_spilling=True,
) as client:
    executor = XennaExecutor(config={
        "execution_mode": "streaming",
    })
    
    # Use bands_per_iteration=1 for minimum memory footprint
    workflow = FuzzyDeduplicationWorkflow(
        input_path="/data/input",
        output_path="/data/output",
        cache_path="/data/cache",
        bands_per_iteration=1,
        # ... other params
    )
    
    results = workflow.run(executor=executor)
```

## Related Skills

- `/setup` - General installation
- `/curate` - Run pipelines
- `/dedup-fuzzy` - Deduplication settings
