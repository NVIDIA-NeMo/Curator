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

The primary NeMo Curator container includes a single conda environment with all necessary dependencies.

(reference-infrastructure-container-environments-curator)=
### Curator Environment

```{list-table} Curator Environment Configuration
:header-rows: 1
:widths: 25 75

* - Property
  - Value
* - Environment Name
  - `curator`
* - Python Version
  - 3.12
* - CUDA Version
  - 12.5.1 (configurable)
* - Operating System
  - Ubuntu 22.04 (configurable)
* - Base Image
  - `rapidsai/ci-conda`
* - Core Dependencies
  - `cuda-cudart`, `libcufft`, `libcublas`, `libcurand`, `libcusparse`, `libcusolver`, `cuda-nvvm`, `pytest`, `pip`, `pytest-coverage`
* - Installation
  - NeMo Curator installed with all optional dependencies (`[all]` extras) from PyPI with NVIDIA index
* - Environment Path
  - Activated by default and added to system PATH: `/opt/conda/envs/curator/bin:$PATH`
```

---

(reference-infrastructure-container-environments-slurm)=
## Slurm Environment Variables

When you deploy NeMo Curator on Slurm clusters, the following environment variables configure the runtime environment:

(reference-infrastructure-container-environments-slurm-defaults)=
### Default Configuration

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `device` | `"cpu"` | Device type: `"cpu"` or `"gpu"` |
| `interface` | `"eth0"` | Network interface for Dask communication |
| `protocol` | `"tcp"` | Network protocol: `"tcp"` or `"ucx"` |
| `cpu_worker_memory_limit` | `"0"` | Memory limit per worker (`"0"` = no limit) |
| `rapids_no_initialize` | `"1"` | Delay CUDA context creation for UCX compatibility |
| `cudf_spill` | `"1"` | Enable automatic GPU memory spilling |
| `rmm_scheduler_pool_size` | `"1GB"` | GPU memory pool size for scheduler |
| `rmm_worker_pool_size` | `"72GiB"` | GPU memory pool size per worker (80–90% of GPU memory) |
| `libcudf_cufile_policy` | `"OFF"` | Direct storage-to-GPU I/O policy |

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

The Slurm configuration automatically generates these additional environment variables:

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
| `CUDA_VER` | `12.5.1` | CUDA version |
| `LINUX_VER` | `ubuntu22.04` | Base OS version |
| `PYTHON_VER` | `3.12` | Python version |
| `IMAGE_LABEL` | - | Container label |
| `REPO_URL` | - | Source repository URL |
| `CURATOR_COMMIT` | - | Git commit to build from |

---

(reference-infrastructure-container-environments-usage)=
## Environment Usage Examples

(reference-infrastructure-container-environments-usage-text)=
### Text Curation
Uses the default `curator` environment with CPU or GPU workers depending on the module.

(reference-infrastructure-container-environments-usage-image)=
### Image Curation  
Requires GPU-enabled workers in the `curator` environment.
