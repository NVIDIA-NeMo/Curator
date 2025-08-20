---
description: "API reference and configuration guide for NeMo Curator's distributed computing functionality using Dask clusters"
categories: ["reference"]
tags: ["distributed", "dask", "clusters", "scaling", "python-api", "configuration"]
personas: ["mle-focused", "admin-focused", "devops-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(reference-infra-dist-computing)=
# Distributed Computing Reference

This reference documents NeMo Curator's distributed computing functionality, which uses Dask to process large datasets across multiple machines.

## API Reference

### `get_client()`

```python
from nemo_curator.utils.distributed_utils import get_client

client = get_client(
    cluster_type="cpu",
    scheduler_address=None,
    scheduler_file=None,
    n_workers=os.cpu_count(),
    threads_per_worker=1,
    nvlink_only=False,
    protocol="tcp",
    rmm_pool_size="1024M",
    enable_spilling=True,
    set_torch_to_use_rmm=False,
    rmm_async=True,
    rmm_maximum_pool_size=None,
    rmm_managed_memory=False,
    rmm_release_threshold=None,
    **cluster_kwargs
)
```

Initializes or connects to a Dask cluster.

**Parameters:**

- `cluster_type`: Either "cpu" or "gpu". Sets up local cluster type if `scheduler_address` and `scheduler_file` are None.
- `scheduler_address`: Address of existing Dask cluster to connect to (e.g., '127.0.0.1:8786').
- `scheduler_file`: Path to a file with scheduler information.
- `n_workers`: (CPU clusters only) Number of workers to start. Defaults to `os.cpu_count()`.
- `threads_per_worker`: (CPU clusters only) Number of threads per worker. Defaults to 1.
- `nvlink_only`: (GPU clusters only) Whether to use NVLink for communication.
- `protocol`: (GPU clusters only) Protocol for communication, "tcp" or "ucx".
- `rmm_pool_size`: (GPU clusters only) RAPIDS Memory Manager pool size per worker.
- `enable_spilling`: (GPU clusters only) Whether to enable automatic memory spilling.
- `set_torch_to_use_rmm`: (GPU clusters only) Whether to use RMM for PyTorch allocations.
- `rmm_async`: (GPU clusters only) Whether to use RMM's asynchronous allocator.
- `rmm_maximum_pool_size`: (GPU clusters only) Maximum pool size for RMM.
- `rmm_managed_memory`: (GPU clusters only) Whether to use CUDA managed memory.
- `rmm_release_threshold`: (GPU clusters only) Threshold for releasing memory from the pool.
- `cluster_kwargs`: Additional keyword arguments for LocalCluster or LocalCUDACluster.

**Returns:**

A Dask client object connected to the specified cluster.

## Client Setup Examples

### Local CPU Cluster

```python
# 8 CPU workers
client = get_client(
    cluster_type="cpu",
    n_workers=8,
    threads_per_worker=1
)
```

### Local GPU Cluster

```python
# One worker per GPU
client = get_client(
    cluster_type="gpu",
    rmm_pool_size="4GB",
    enable_spilling=True
)
```

### Connect to Existing Cluster

```python
# Connect to scheduler
client = get_client(
    scheduler_address="tcp://scheduler-address:8786"
)

# Using scheduler file
client = get_client(
    scheduler_file="/path/to/scheduler.json"
)
```

## Partition Control

Control how data is partitioned across workers:

```python
from nemo_curator.datasets import DocumentDataset

# Adjust partition size based on cluster resources
dataset = DocumentDataset.read_json(
    files,
    blocksize="1GB",  # Size per partition
    files_per_partition=100  # Files per partition
)
```

## Resource Management

Monitor and manage cluster resources:

```python
# Access dashboard
print(client.dashboard_link)

# Get worker memory information
worker_memory = client.get_worker_logs()

# Restart workers if needed
client.restart()
```
