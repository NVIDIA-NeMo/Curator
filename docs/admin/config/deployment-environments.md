---
description: "Configure NeMo Curator for different deployment environments including local development, Slurm clusters, and Kubernetes"
categories: ["how-to-guides"]
tags: ["deployment-environments", "slurm", "kubernetes", "ray-clusters", "gpu-settings", "networking", "performance"]
personas: ["admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "universal"
---

(admin-config-deployment-environments)=

# Deployment Environment Configuration

Configure NeMo Curator for different deployment environments including local development, Slurm clusters, and Kubernetes. This guide focuses on deployment-specific settings and operational concerns for Ray-based deployments.

```{tip}
**Applying These Configurations**: This guide shows you how to configure NeMo Curator for different environments. To learn how to actually deploy and run NeMo Curator in these environments, see:
- {doc}`Kubernetes Deployment <../deployment/kubernetes>`: Running on Kubernetes clusters
- {doc}`Slurm Deployment <../deployment/slurm/index>`: Running on Slurm-managed clusters  
- {doc}`Deployment Options <../deployment/index>`: Overview of all deployment methods
```

---

## Deployment Scenarios

### Local Development Environment

Basic configuration for single-machine development and testing.

::::{tab-set}

:::{tab-item} CPU-Only Setup
:sync: local-cpu

```bash
# Environment variables for local CPU development
export RAY_NUM_CPUS="4"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE="1"
export NEMO_CURATOR_LOG_LEVEL="INFO"
export NEMO_CURATOR_CACHE_DIR="./cache"
```
:::

:::{tab-item} GPU Development Setup
:sync: local-gpu

```bash
# Environment variables for local GPU development
export RAY_NUM_GPUS="1"
export RMM_WORKER_POOL_SIZE="4GB"
export CUDF_SPILL="1"
export NEMO_CURATOR_LOG_LEVEL="DEBUG"
```
:::

::::

### Production Slurm Environment

Optimized configuration for Slurm-managed GPU clusters.

::::{tab-set}

:::{tab-item} Standard Configuration
:sync: slurm-standard

```bash
# Production Slurm environment variables
export DEVICE="gpu"
export PROTOCOL="ucx"  # Use UCX for multi-GPU communication
export INTERFACE="ib0"  # InfiniBand interface if available
export CPU_WORKER_MEMORY_LIMIT="0"  # No memory limit
export RAPIDS_NO_INITIALIZE="0"
export CUDF_SPILL="0"  # Disable spilling for performance
export RMM_SCHEDULER_POOL_SIZE="1GB"
export RMM_WORKER_POOL_SIZE="80GiB"  # 80-90% of GPU memory
export LIBCUDF_CUFILE_POLICY="ON"  # Enable GPUDirect Storage
```
:::

:::{tab-item} High-Performance Setup
:sync: slurm-highperf

```bash
# High-performance Slurm configuration
export DEVICE="gpu"
export PROTOCOL="ucx"
export INTERFACE="ib0"
export UCX_MEMTYPE_CACHE="n"  # Disable UCX memory type cache
export UCX_TLS="rc,cuda_copy,cuda_ipc"  # Optimized transport layers
export RMM_WORKER_POOL_SIZE="90GiB"  # Maximum GPU memory allocation
export CUDF_SPILL="0"
export LIBCUDF_CUFILE_POLICY="ON"
export NEMO_CURATOR_LOG_LEVEL="WARNING"  # Reduce logging overhead
```

For optimal performance on large clusters.
:::

::::

### Kubernetes Environment

Configuration for Kubernetes deployments with Ray Operator.

::::{tab-set}

:::{tab-item} Basic Setup
:sync: k8s-basic

```yaml
# kubernetes-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nemo-curator-config
data:
  RAY_ADDRESS: "ray://ray-head:10001"
  RMM_WORKER_POOL_SIZE: "16GB"
  CUDF_SPILL: "1"
  NEMO_CURATOR_LOG_LEVEL: "INFO"
  NEMO_CURATOR_CACHE_DIR: "/shared/cache"
```
:::

:::{tab-item} GPU-Enabled
:sync: k8s-gpu

```yaml
# gpu-kubernetes-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nemo-curator-gpu-config
data:
  DEVICE: "gpu"
  PROTOCOL: "ucx"
  RMM_WORKER_POOL_SIZE: "32GB"
  CUDF_SPILL: "0"
  RAPIDS_NO_INITIALIZE: "0"
  LIBCUDF_CUFILE_POLICY: "OFF"  # Usually not available in K8s
  NEMO_CURATOR_LOG_LEVEL: "WARNING"
```
:::

::::

---

## Ray Cluster Configuration

### Cluster Connection Methods

::::{tab-set}

:::{tab-item} Existing Cluster
:sync: cluster-existing

```python
import ray
from nemo_curator.core.client import RayClient

# Connect to existing Ray cluster
ray.init(address="ray://head-node:10001")

# Connect using RAY_ADDRESS environment variable
ray.init()  # Uses RAY_ADDRESS if set
```
:::

:::{tab-item} Local Cluster Creation
:sync: cluster-local

```python
# Create local Ray cluster
ray_client = RayClient(
    num_cpus=4,
    ray_temp_dir="./ray_temp"
)
ray_client.start()

# Create local GPU cluster
ray_client = RayClient(
    num_gpus=1,
    num_cpus=4,
    enable_object_spilling=True
)
ray_client.start()
```
:::

::::

### Cluster Sizing Guidelines

```{list-table} Recommended Cluster Configurations
:header-rows: 1
:widths: 25 25 25 25

* - Use Case
  - Workers
  - Memory per Worker
  - GPU Memory Pool
* - Development
  - 1-2
  - 4-8 GB
  - 2-4 GB
* - Small Production
  - 4-8
  - 16-32 GB
  - 16-32 GB
* - Large Production
  - 16-64
  - 32-128 GB
  - 64-90 GB
* - Massive Scale
  - 64+
  - 128+ GB
  - 80-90 GB
```

---

## GPU Memory Management

### RMM Pool Configuration

Configure RAPIDS Memory Manager for optimal GPU memory usage:

::::{tab-set}

:::{tab-item} Conservative Setup
:sync: rmm-conservative

```bash
# Conservative setup (development)
export RMM_WORKER_POOL_SIZE="4GB"
export CUDF_SPILL="1"
```

Recommended for development and testing environments.
:::

:::{tab-item} Aggressive Setup
:sync: rmm-aggressive

```bash
# Aggressive setup (production)
export RMM_WORKER_POOL_SIZE="80GiB"  # 80-90% of GPU memory
export CUDF_SPILL="0"  # Disable spilling for performance
```

Optimized for production environments with dedicated GPU resources.
:::

::::

### Memory Pool Sizing

```{list-table} RMM Pool Sizing Guidelines
:header-rows: 1
:widths: 25 25 25 25

* - GPU Memory
  - Conservative Pool
  - Balanced Pool
  - Aggressive Pool
* - 16 GB
  - 8 GB
  - 12 GB
  - 14 GB
* - 32 GB
  - 16 GB
  - 24 GB
  - 28 GB
* - 80 GB (A100)
  - 40 GB
  - 64 GB
  - 72 GB
* - 128 GB (H100)
  - 64 GB
  - 96 GB
  - 115 GB
```

---

## Networking Configuration

### Protocol Selection

```{list-table} Network Protocol Recommendations
:header-rows: 1
:widths: 30 20 25 25

* - Deployment Type
  - Recommended Protocol
  - Performance
  - Requirements
* - Single Machine
  - TCP
  - Good
  - None
* - Multi-Node CPU
  - TCP
  - Good
  - Standard networking
* - Multi-Node GPU
  - UCX
  - Excellent
  - UCX-enabled cluster
* - InfiniBand Cluster
  - UCX
  - Excellent
  - InfiniBand + UCX
```

### Network Interface Selection

```bash
# Ethernet (most common)
export INTERFACE="eth0"

# InfiniBand (high-performance clusters)
export INTERFACE="ib0"

# Auto-detect (let Dask choose)
export INTERFACE=""  # Empty string for auto-detection
```

---

## Logging and Monitoring

### Deployment-Specific Logging

::::{tab-set}

:::{tab-item} Development Logging
:sync: logging-dev

```bash
export NEMO_CURATOR_LOG_LEVEL="DEBUG"
export NEMO_CURATOR_LOG_DIR="./logs"
export RAY_LOG_TO_STDERR="1"
```
:::

:::{tab-item} Production Logging
:sync: logging-prod

```bash
export NEMO_CURATOR_LOG_LEVEL="WARNING"
export NEMO_CURATOR_LOG_DIR="/shared/logs"
export RAY_LOG_TO_STDERR="0"
```
:::

::::

### Log Directory Structure

```bash
# Typical production log structure
/shared/logs/
├── ray-head.log          # Ray head node logs
├── ray-worker-*.log      # Individual worker logs
├── nemo-curator.log      # Application logs
└── performance/          # Performance profiles
    ├── ray-timeline.json
    └── ray-profile.html
```

---

## Environment-Specific Optimizations

::::{tab-set}

:::{tab-item} Slurm-Specific Settings
:sync: env-slurm

```bash
# Slurm job integration
export SLURM_JOB_ID="${SLURM_JOB_ID}"
export LOGDIR="${SLURM_SUBMIT_DIR}/logs"
export SCHEDULER_FILE="${LOGDIR}/scheduler.json"

# Slurm-aware resource allocation
export RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK}"
export RAY_NUM_GPUS="${SLURM_GPUS_PER_NODE}"
export NEMO_CURATOR_RAY_SLURM_JOB="1"
```
:::

:::{tab-item} Kubernetes-Specific Settings
:sync: env-k8s

```bash
# Kubernetes pod integration
export K8S_NAMESPACE="${MY_POD_NAMESPACE}"
export K8S_POD_NAME="${MY_POD_NAME}"
export RAY_ADDRESS="ray://ray-head:10001"

# Kubernetes resource limits
export RAY_ADDRESS="ray://ray-head:10001"
export RMM_WORKER_POOL_SIZE="${GPU_MEMORY_LIMIT}"
```
:::

::::

---

## Validation and Testing

::::{tab-set}

:::{tab-item} Cluster Connectivity Test
:sync: test-cluster

```python
import ray

# Test Ray cluster connection
ray.init()
print(f"✓ Connected to Ray cluster")
print(f"✓ Nodes: {len(ray.nodes())}")
print(f"✓ Dashboard: http://localhost:8265")
```
:::

:::{tab-item} GPU Configuration Test
:sync: test-gpu

```python
# Test GPU availability and configuration
try:
    import cudf
    df = cudf.DataFrame({"test": [1, 2, 3]})
    print("✓ GPU processing available")
    
    # Test RMM configuration
    import rmm
    print(f"✓ RMM pool size: {rmm.get_current_device_resource()}")
except ImportError as e:
    print(f"⚠ GPU not available: {e}")
```
:::

::::
