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

This reference documents NeMo Curator's distributed computing functionality. NeMo Curator uses a Ray-based execution engine to process large datasets on one or more machines.

## API Reference

### Start a local Ray cluster

```python
from nemo_curator.core.client import RayClient

# Start a local Ray cluster with optional resource hints
ray_client = RayClient(
    num_gpus=1,           # or None to auto-detect
    num_cpus=None,        # defaults to system CPU count
    include_dashboard=True,
)

ray_client.start()

# ... run NeMo Curator pipelines ...

# When finished
ray_client.stop()
```

Initializes or connects to a local Ray cluster and optionally configures metrics integration.

**Key parameters:**

- `num_gpus`: Number of GPU devices. If `None`, Ray detects available GPU devices.
- `num_cpus`: Number of CPU cores. If `None`, Ray detects available CPU cores.
- `include_dashboard`: Integrate with Prometheus/Grafana if available.

---

## Running pipelines on Ray

NeMo Curator stages and pipelines run on Ray once `RayClient.start()` has initialized a cluster. Configure stage resources using `Resources(cpus=..., gpus=...)` where supported.

## Partition Control

Control how workers partition the data:

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

Manage cluster resources:

```python
# Access dashboard
print(client.dashboard_link)

# Get worker memory information
worker_memory = client.get_worker_logs()

# Restart workers if needed
client.restart()
```
