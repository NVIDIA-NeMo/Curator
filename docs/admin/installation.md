---
description: "Complete installation guide for NeMo Curator with system requirements, package extras, verification steps, and troubleshooting"
categories: ["getting-started"]
tags: ["installation", "system-requirements", "pypi", "source-install", "container", "verification", "troubleshooting"]
personas: ["admin-focused", "devops-focused", "data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "universal"
---

(admin-installation)=
# Installation Guide

This guide covers installing NeMo Curator and verifying your installation is working correctly. For configuration after installation, see [Configuration](admin-config).

## System Requirements

For comprehensive system requirements and production deployment specifications, see [Production Deployment Requirements](deployment/requirements.md).

**Quick Start Requirements:**
- **OS**: Ubuntu 22.04/20.04 (recommended) 
- **Python**: 3.10 or 3.12 (Python 3.11 is not supported)
- **Memory**: 16GB+ RAM for basic text processing
- **GPU** (optional): NVIDIA GPU with 16GB+ VRAM for acceleration

### Development vs Production

| Use Case | Requirements | See |
|----------|-------------|-----|
| **Local Development** | Minimum specs listed above | Continue below |
| **Production Clusters** | Detailed hardware, network, storage specs | [Deployment Requirements](deployment/requirements.md) |
| **Multi-node Setup** | Advanced infrastructure planning | [Deployment Options](deployment/index.md) |

---

## Installation Methods

Choose one of the following installation methods based on your needs:

::::{tab-set}

:::{tab-item} PyPI Installation (Recommended)

The simplest way to install NeMo Curator from the Python Package Index:

**CPU-only installation:**
```bash
pip install nemo-curator
```

**GPU-accelerated installation:**
```bash
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[cuda12x]
```

**Full installation with all modules:**
```bash
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[all]
```

:::

:::{tab-item} Source Installation

Install the latest development version directly from GitHub:

```bash
# Clone the repository
git clone https://github.com/NVIDIA/NeMo-Curator.git
cd NeMo-Curator

# Install with desired extras
pip install --extra-index-url https://pypi.nvidia.com ".[all]"
```

**Benefits:**
- Access to latest features and bug fixes
- Ability to modify source code for custom needs
- Easier contribution to the project

:::

:::{tab-item} Container Installation

NeMo Curator is available as a standalone container:

```{warning}
**Container Availability**: The standalone NeMo Curator container is currently in development. Check the [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers) for the latest availability and container path.
```

```bash
# Pull the container (path will be updated when available)
docker pull nvcr.io/nvidia/nemo-curator:latest

# Run the container with GPU support
docker run --gpus all -it --rm nvcr.io/nvidia/nemo-curator:latest

# For custom installations inside container
pip uninstall nemo-curator
rm -r /opt/NeMo-Curator
git clone https://github.com/NVIDIA/NeMo-Curator.git /opt/NeMo-Curator
pip install --extra-index-url https://pypi.nvidia.com "/opt/NeMo-Curator[all]"
```

**Benefits:**
- Pre-configured environment with all dependencies
- Consistent runtime across different systems
- Ideal for production deployments

:::

::::

---

## Package Extras

NeMo Curator provides several installation extras to install only the components you need:

```{list-table} Available Package Extras
:header-rows: 1
:widths: 20 30 50

* - Extra
  - Installation Command
  - Description
* - **Base**
  - `pip install nemo-curator`
  - CPU-only text curation modules
* - **dev**
  - `pip install nemo-curator[dev]`
  - Development tools (pre-commit, ruff, pytest)
* - **cuda12x**
  - `pip install --extra-index-url https://pypi.nvidia.com nemo-curator[cuda12x]`
  - CPU + GPU text curation with RAPIDS
* - **audio_cpu**
  - `pip install nemo-curator[audio_cpu]`
  - CPU-only audio curation with NeMo Toolkit ASR
* - **audio_cuda12**
  - `pip install --extra-index-url https://pypi.nvidia.com nemo-curator[audio_cuda12]`
  - GPU-accelerated audio curation with NeMo Toolkit ASR
* - **image**
  - `pip install --extra-index-url https://pypi.nvidia.com nemo-curator[image]`
  - CPU + GPU text and image curation
* - **all**
  - `pip install --extra-index-url https://pypi.nvidia.com nemo-curator[all]`
  - All stable modules (recommended)
```

### Nightly Dependencies

For cutting-edge RAPIDS features, use nightly builds:

```bash
# Nightly RAPIDS with all modules
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[all_nightly]

# Nightly RAPIDS with image modules
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[image_nightly]
```

```{warning}
Nightly builds may be unstable and are not recommended for production use.
```

---

## Installation Verification

After installation, verify that NeMo Curator is working correctly:

### 1. Basic Import Test

```python
# Test basic imports
import nemo_curator
print(f"NeMo Curator version: {nemo_curator.__version__}")

# Test core modules
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates
print("✓ Core modules imported successfully")
```

### 2. GPU Availability Check

If you installed GPU support, verify GPU access:

```python
# Check GPU availability
try:
    import cudf
    import dask_cudf
    print("✓ GPU modules available")
    
    # Test GPU memory
    import cupy
    mempool = cupy.get_default_memory_pool()
    print(f"✓ GPU memory pool initialized: {mempool.total_bytes() / 1e9:.1f} GB")
except ImportError as e:
    print(f"⚠ GPU modules not available: {e}")
```

### 3. CLI Tools Verification

Test that command-line tools are properly installed:

```bash
# Check if CLI tools are available
text_cleaning --help
add_id --help
gpu_exact_dups --help

# Test specific functionality
echo '{"id": "doc1", "text": "Hello world"}' | text_cleaning --input-format jsonl
```

### 4. Dask Cluster Test

Verify distributed computing capabilities:

```python
from nemo_curator.utils.distributed_utils import get_client

# Test local cluster creation
client = get_client(cluster_type="local", n_workers=2)
print(f"✓ Dask cluster created: {client}")

# Test basic distributed operation
import dask.dataframe as dd
df = dd.from_pandas(pd.DataFrame({"x": [1, 2, 3, 4]}), npartitions=2)
result = df.x.sum().compute()
print(f"✓ Distributed computation successful: {result}")

client.close()
```

---

## Common Installation Issues

### CUDA/GPU Issues

**Problem**: GPU modules not available after installation
```bash
ImportError: No module named 'cudf'
```

**Solutions**:
1. Ensure you installed with the correct extra: `nemo-curator[cuda12x]` or `nemo-curator[all]`
2. Verify CUDA is properly installed: `nvidia-smi`
3. Check CUDA version compatibility (CUDA 12.0+ required)
4. Install RAPIDS manually: `pip install --extra-index-url https://pypi.nvidia.com cudf-cu12`

### Python Version Issues

**Problem**: Installation fails with Python version errors
```bash
ERROR: Package 'nemo_curator' requires a different Python: 3.9.0 not in '>=3.10'
```

**Solutions**:
1. Upgrade to Python 3.10 or 3.12
2. Use conda to manage Python versions: `conda create -n curator python=3.12`
3. Avoid Python 3.11 (not supported due to RAPIDS compatibility)

### Network/Registry Issues

**Problem**: Cannot access NVIDIA PyPI registry
```bash
ERROR: Could not find a version that satisfies the requirement cudf-cu12
```

**Solutions**:
1. Ensure you're using the NVIDIA registry: `--extra-index-url https://pypi.nvidia.com`
2. Check network connectivity to PyPI and NVIDIA registry
3. Try installing with `--trusted-host pypi.nvidia.com`
4. Use container installation as alternative

### Memory Issues

**Problem**: Installation fails due to insufficient memory
```bash
MemoryError: Unable to allocate array
```

**Solutions**:
1. Increase system memory or swap space
2. Install packages individually rather than `[all]`
3. Use `--no-cache-dir` flag: `pip install --no-cache-dir nemo-curator[all]`
4. Consider container installation

---

## Next Steps

Choose your next step based on your goals:

### For Local Development & Learning
1. **Try a tutorial**: Start with [Get Started guides](../get-started/index.md)
2. **Configure your environment**: See [Configuration Guide](config/index.md) for basic setup

### For Production Deployment
1. **Review requirements**: See [Production Deployment Requirements](deployment/requirements.md)
2. **Choose deployment method**: See [Deployment Options](deployment/index.md)
3. **Configure for production**: See [Configuration Guide](config/index.md) for advanced settings

```{seealso}
- [Configuration Guide](config/index.md) - Configure NeMo Curator for your environment
- [Container Environments](../reference/infrastructure/container-environments.md) - Container-specific setup
- [Deployment Requirements](deployment/requirements.md) - Production deployment prerequisites
``` 