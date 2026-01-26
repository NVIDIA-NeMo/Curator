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

```python
import ray

# Start Ray locally (uses all CPUs and GPUs)
ray.init()

# Or with explicit resources
ray.init(num_cpus=8, num_gpus=2)
```

### Multi-Node Cluster

```bash
# On head node
ray start --head --port=6379

# On worker nodes
ray start --address='<head-ip>:6379'
```

## Ray Configuration in NeMo Curator

### Via Pipeline YAML

```yaml
ray_client:
  _target_: nemo_curator.core.client.RayClient
  num_cpus: null  # null = use all available
  num_gpus: 4
```

### Via Python

```python
from nemo_curator.core.client import RayClient

client = RayClient(
    num_cpus=None,  # All available
    num_gpus=4,
)
```

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

NeMo Curator stages declare GPU requirements:

```python
from nemo_curator.stages.resources import Resources

# Stage with shared GPU (memory fraction)
Resources(gpu_memory_gb=16.0)  # Uses portion of GPU

# Stage with dedicated GPU
Resources(gpus=1.0)  # Reserves entire GPU
```

### Memory Configuration

For memory-intensive workloads:

```python
ray.init(
    _memory=200 * 1024**3,  # 200GB heap
    object_store_memory=100 * 1024**3,  # 100GB object store
)
```

## Executor Configuration

### XennaExecutor (Default)

```python
from nemo_curator.backends.xenna import XennaExecutor

executor = XennaExecutor(config={
    "logging_interval": 60,           # Log every 60 seconds
    "ignore_failures": False,         # Stop on errors
    "execution_mode": "streaming",    # For large datasets
    "cpu_allocation_percentage": 0.95,
    "autoscale_interval_s": 180,      # Autoscaling interval
})
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

```python
import ray
ray.init(address='auto')  # Connect to existing cluster

print(f"Nodes: {len(ray.nodes())}")
print(f"Total CPUs: {ray.cluster_resources().get('CPU', 0)}")
print(f"Total GPUs: {ray.cluster_resources().get('GPU', 0)}")
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

For deduplication (shuffle-heavy):

```python
# Increase shuffle partitions for large data
ray.init(
    _system_config={
        "max_pending_lease_requests_per_scheduling_category": 100,
    }
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

```python
# For memory-intensive stages
ray.init(
    object_store_memory=100 * 1024**3,  # 100GB
    _plasma_directory="/dev/shm",  # Use tmpfs
)
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

```python
# Enable detailed logging
ray.init(
    logging_level="DEBUG",
    log_to_driver=True,
)
```

## Example Configurations

### Development (Single Node)

```python
ray.init(num_cpus=8, num_gpus=1)

executor = XennaExecutor(config={
    "execution_mode": "batch",
})
```

### Production (Multi-Node)

```python
ray.init(address="auto")  # Connect to existing cluster

executor = XennaExecutor(config={
    "execution_mode": "streaming",
    "cpu_allocation_percentage": 0.95,
    "autoscale_interval_s": 180,
    "ignore_failures": False,
})
```

### Large-Scale Deduplication

```python
ray.init(
    address="auto",
    _memory=200 * 1024**3,
    object_store_memory=100 * 1024**3,
)

executor = XennaExecutor(config={
    "execution_mode": "streaming",
})

# Use bands_per_iteration=1 in FuzzyDeduplicationWorkflow
```

## Related Skills

- `/setup` - General installation
- `/setup-verify` - Verify installation
- `/curate` - Run pipelines
- `/dedup-fuzzy` - Deduplication settings
