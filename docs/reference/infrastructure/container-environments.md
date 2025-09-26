---
description: "Reference documentation for container environments, configurations, and deployment variables in NeMo Curator"
categories: ["reference"]
tags: ["docker", "slurm", "kubernetes", "configuration", "deployment", "gpu-accelerated", "environments"]
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

(reference-infrastructure-container-environments-slurm)=

## Slurm Environment Variables

When you deploy NeMo Curator on Slurm clusters, the following environment variables configure the runtime environment:

(reference-infrastructure-container-environments-slurm-defaults)=

### Default Configuration

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `DEVICE` | `"cpu"` | Device type: `"cpu"` or `"gpu"` |
| `INTERFACE` | `"eth0"` | Network interface for Dask communication |
| `PROTOCOL` | `"tcp"` | Network protocol: `"tcp"` or `"ucx"` |
| `CPU_WORKER_MEMORY_LIMIT` | `"0"` | Memory limit per worker (`"0"` = no limit) |
| `RAPIDS_NO_INITIALIZE` | `"1"` | Delay CUDA context creation for UCX compatibility |
| `CUDF_SPILL` | `"1"` | Enable automatic GPU memory spilling |
| `RMM_SCHEDULER_POOL_SIZE` | `"1GB"` | GPU memory pool size for scheduler |
| `RMM_WORKER_POOL_SIZE` | `"72GiB"` | GPU memory pool size per worker (80–90% of GPU memory) |
| `LIBCUDF_CUFILE_POLICY` | `"OFF"` | Direct storage-to-GPU I/O policy |

(reference-infrastructure-container-environments-slurm-gpu)=

### GPU Configuration Recommendations

For GPU workloads, consider these optimized settings:

```bash
export DEVICE="gpu"
export PROTOCOL="ucx"  # If your cluster supports it
export INTERFACE="ib0"  # If you're using InfiniBand
export RAPIDS_NO_INITIALIZE="0"
export CUDF_SPILL="0"
export RMM_WORKER_POOL_SIZE="80GiB"  # Adjust based on your GPU memory
export LIBCUDF_CUFILE_POLICY="ON"  # If GPUDirect Storage is available
```

(reference-infrastructure-container-environments-slurm-auto)=

### Automatic Environment Variables

The Slurm configuration automatically generates these environment variables:

| Variable | Value | Description |
|----------|-------|-------------|
| `LOGDIR` | `{job_dir}/logs` | Directory for Dask logs |
| `PROFILESDIR` | `{job_dir}/profiles` | Directory for performance profiles |
| `SCHEDULER_FILE` | `{LOGDIR}/scheduler.json` | Dask scheduler connection file |
| `SCHEDULER_LOG` | `{LOGDIR}/scheduler.log` | Scheduler log file |
| `DONE_MARKER` | `{LOGDIR}/done.txt` | Job completion marker |

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
