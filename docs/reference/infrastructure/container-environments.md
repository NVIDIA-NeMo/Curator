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

This reference documents the default environments available in NeMo Curator containers and their configurations.

(reference-infrastructure-container-environments-main)=

## Main Container Environment

The primary NeMo Curator container includes a uv-managed virtual environment with all necessary dependencies.

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
