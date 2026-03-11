---
description: "Comprehensive system, hardware, and software requirements for deploying NeMo Curator in production environments"
categories: ["reference"]
tags: ["requirements", "system-requirements", "hardware", "software", "gpu", "storage"]
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
- **Python**: Python 3.10, 3.11, or 3.12
  - packaging >= 22.0

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
- Python 3.10+ with required packages for distributed computing
- RAPIDS libraries (cuDF) for GPU-accelerated deduplication operations

### Container Support (Recommended)
- **Docker** or **Podman** for containerized deployment
- Access to NVIDIA NGC registry for official containers


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

- Resource quotas configured for GPU and memory allocation

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