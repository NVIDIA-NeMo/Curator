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

# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with all extras using uv
uv sync --extra all
```

**Benefits:**
- Access to latest features and bug fixes
- Ability to modify source code for custom needs
- Faster dependency resolution with uv
- Easier contribution to the project

:::

:::{tab-item} Container Installation

NeMo Curator is available as a standalone container:

```{note}
**Container Build**: You can build the NeMo Curator container locally using the provided Dockerfile. A pre-built container will be available on NGC in the future.
```

```bash
# Build the container locally
git clone https://github.com/NVIDIA/NeMo-Curator.git
cd NeMo-Curator
docker build -t nemo-curator:latest -f docker/Dockerfile .

# Run the container with GPU support
docker run --gpus all -it --rm nemo-curator:latest

# The container includes NeMo Curator with all dependencies pre-installed
# Environment is activated automatically at /opt/venv
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

---

## Installation Verification

After installation, verify that NeMo Curator is working correctly:

### 1. Basic Import Test

```python
# Test basic imports
import nemo_curator
print(f"NeMo Curator version: {nemo_curator.__version__}")

# Test core modules
from nemo_curator.pipeline import Pipeline
from nemo_curator.tasks import DocumentBatch
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

### 3. Module Import Verification

Test that core modules can be imported successfully:

```python
# Test core processing modules
from nemo_curator.stages.text.modules.add_id import AddId
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter
print("✓ Core processing modules imported successfully")

# Test pipeline components
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.backends.base import BaseExecutor
print("✓ Pipeline components imported successfully")
```

### 4. Ray Cluster Test

Verify distributed computing capabilities:

```python
import ray
from nemo_curator.core.client import RayClient

# Test Ray cluster creation
ray_client = RayClient(num_cpus=2, num_gpus=0)
ray_client.start()
print("✓ Ray cluster created successfully")

# Test basic distributed operation
import pandas as pd
@ray.remote
def sum_data(data):
    return sum(data)

# Simple distributed computation test
future = sum_data.remote([1, 2, 3, 4])
result = ray.get(future)
print(f"✓ Distributed computation successful: {result}")

ray_client.stop()
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
2. Use virtual environments to manage Python versions: `python3.12 -m venv curator-env`
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