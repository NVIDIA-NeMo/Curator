---
description: "Reference documentation for container environments, configurations, and deployment variables in NeMo Curator"
categories: ["reference"]
tags: ["docker", "configuration", "deployment", "gpu-accelerated", "environments"]
personas: ["admin-focused", "devops-focused", "mle-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(reference-infrastructure-container-environments)=

# Container Environments

Deploy NeMo Curator in containerized environments for reproducible, scalable data curation pipelines with pre-configured dependencies and optimized runtime settings.

## Overview

NeMo Curator provides official Docker containers with all dependencies pre-installed and optimized for production workloads. Containers offer:

- **Reproducible Environments**: Consistent software stack across development, testing, and production
- **Simplified Deployment**: No manual dependency installation or environment configuration
- **GPU Acceleration**: Pre-configured CUDA, cuDNN, and NVIDIA libraries for optimal performance
- **Multi-Modal Support**: Built-in support for text, image, video, and audio curation
- **Cloud-Ready**: Compatible with Kubernetes, Docker Swarm, and cloud container orchestries

**When to use containers:**
- Production deployments requiring consistency and reliability
- Multi-node cluster processing with identical environments
- CI/CD pipelines for automated data curation workflows
- Quick prototyping without local environment setup
- GPU-accelerated processing in cloud environments

## Available Containers

### Main NeMo Curator Container

The primary container includes comprehensive support for all curation modalities:

**Container registry:** `nvcr.io/nvidia/nemo-curator:25.09`

**Supported modalities:**
- ✅ Text curation (CPU/GPU)
- ✅ Image curation (GPU required)
- ✅ Video curation (GPU required, FFmpeg included)
- ✅ Audio curation (GPU required for ASR)

**Pre-installed components:**
- NeMo Curator with all optional dependencies (`[all]` extras)
- CUDA 12.8.1 with cuDNN
- Python 3.12 with uv package manager
- FFmpeg 8+ with NVENC support (for video processing)
- Ray, Dask, and distributed computing frameworks
- NVIDIA optimized Python packages

(reference-infrastructure-container-environments-curator)=

### Curator Environment

```{list-table} Curator Environment Configuration
:header-rows: 1
:widths: 25 75

* - Property
  - Value
* - Python Version
  - 3.12
* - CUDA Version
  - 12.8.1 (configurable)
* - Operating System
  - Ubuntu 24.04 (configurable)
* - Base Image
  - `nvidia/cuda:${CUDA_VER}-cudnn-devel-${LINUX_VER}`
* - Package Manager
  - uv (Ultrafast Python package installer)
* - Installation
  - NeMo Curator installed with all optional dependencies (`[all]` extras) using uv with NVIDIA index
* - Environment Path
  - Virtual environment activated by default: `/opt/venv/bin:$PATH`
```

---

(reference-infrastructure-container-environments-build-args)=

## Container Build Arguments

The main container accepts these build-time arguments for environment customization:

| Argument | Default | Description |
|----------|---------|-------------|
| `CUDA_VER` | `12.8.1` | CUDA version |
| `LINUX_VER` | `ubuntu24.04` | Base OS version |
| `CURATOR_ENV` | `ci` | Curator environment type |
| `INTERN_VIDEO_COMMIT` | `09d872e5...` | InternVideo commit hash for video curation |
| `NVIDIA_BUILD_ID` | `<unknown>` | NVIDIA build identifier |
| `NVIDIA_BUILD_REF` | - | NVIDIA build reference |

---

(reference-infrastructure-container-environments-usage)=

## Environment Usage Examples

(reference-infrastructure-container-environments-usage-text)=

### Text Curation

Uses the default container environment with CPU or GPU workers depending on the module.

(reference-infrastructure-container-environments-usage-image)=

### Image Curation

Requires GPU-enabled workers in the container environment.
