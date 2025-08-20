---
description: "Comprehensive system, hardware, and software requirements for deploying NeMo Curator in production environments"
categories: ["reference"]
tags: ["requirements", "system-requirements", "hardware", "software", "kubernetes", "slurm", "gpu", "storage"]
personas: ["admin-focused", "devops-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(admin-deployment-requirements)=
# Production Deployment Requirements

This page details the comprehensive system, hardware, and software requirements for deploying NeMo Curator in production environments.

## System Requirements

- **Operating System**: Ubuntu 22.04/20.04 (recommended)
- **Python**: Python 3.10 or 3.12 (Python 3.11 is not supported due to RAPIDS compatibility)
  - packaging >= 22.0
- **Shared Filesystem**: For Slurm deployments, a shared filesystem (NFS, Lustre, etc.) accessible from all compute nodes

## Hardware Requirements

### CPU Requirements
- Multi-core CPU with sufficient cores for parallel processing
- **Memory**: Minimum 16GB RAM recommended for text processing
  - For large datasets: 32GB+ RAM recommended
  - Memory requirements scale with dataset size and number of workers

### GPU Requirements (Optional but Recommended)
- **GPU**: NVIDIA GPU with Volta™ architecture or higher
  - Compute capability 7.0+ required
  - **Memory**: Minimum 16GB VRAM for GPU-accelerated operations
  - For video processing: 21GB+ VRAM (reducible with optimization)
  - For large-scale deduplication: 32GB+ VRAM recommended
- **CUDA**: CUDA 12.0 or above with compatible drivers

## Software Dependencies

### Core Dependencies
- [Dask](https://docs.dask.org/en/stable/) for distributed computing
- [dask-cuda](https://docs.rapids.ai/api/dask-cuda/stable/) for GPU-enabled clusters
- RAPIDS libraries (cuDF, cuML, cuGraph) for GPU acceleration

### Container Support (Recommended)
- **Docker** or **Podman** for containerized deployment
- **Singularity/Apptainer** for HPC environments
- Access to NVIDIA NGC registry for official containers

### Cluster Management
- **Kubernetes**: For Kubernetes deployment
  - [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html)
  - [Dask Operator](https://kubernetes.dask.org/en/latest/operator_installation.html)
  - [kubectl](https://kubernetes.io/docs/tasks/tools) configured with cluster access
  - ReadWriteMany StorageClass for shared storage
- **Slurm**: For Slurm deployment
  - Slurm cluster with job submission permissions
  - Shared filesystem mounted on all compute nodes

## Network Requirements
- Reliable network connectivity between nodes
- High-bandwidth network for large dataset transfers
- InfiniBand recommended for multi-node GPU clusters

## Storage Requirements
- **Capacity**: Storage capacity should be 3-5x the size of input datasets
  - Input data storage
  - Intermediate processing files
  - Output data storage
- **Performance**: High-throughput storage system recommended
  - SSD storage preferred for frequently accessed data
  - Parallel filesystem for multi-node access

## Deployment-Specific Requirements

### Kubernetes Deployment
- Kubernetes cluster with GPU support
- Persistent Volume Claims (PVC) with ReadWriteMany access mode
- Network policies allowing inter-pod communication
- Resource quotas configured for GPU and memory allocation

### Slurm Deployment
- Slurm workload manager configured and running
- Job submission permissions (`sbatch`, `srun`)
- Module system for environment management (optional but recommended)
- Batch job script templates and container runtime support

## Performance Considerations

### Memory Management
- Monitor memory usage across distributed workers
- Configure appropriate memory limits per worker
- Use memory-efficient data formats (e.g., Parquet)

### GPU Optimization
- Ensure CUDA drivers are compatible with RAPIDS versions
- Configure GPU memory pools (RMM) for optimal performance
- Monitor GPU utilization and memory usage

### Network Optimization
- Use high-bandwidth interconnects for multi-node deployments
- Configure appropriate network protocols (TCP vs UCX)
- Optimize data transfer patterns to minimize network overhead 