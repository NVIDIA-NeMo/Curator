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

This guide covers installing NeMo Curator with support for **all modalities** and verifying your installation is working correctly.

## Before You Start

### System Requirements

For comprehensive system requirements and production deployment specifications, see [Production Deployment Requirements](deployment/requirements.md).

**Quick Start Requirements:**

- **OS**: Ubuntu 24.04/22.04/20.04 (recommended)
- **Python**: 3.10, 3.11, or 3.12
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

Install NeMo Curator from the Python Package Index using `uv` for proper dependency resolution.

1. Install uv:

   ```bash
   curl -LsSf https://astral.sh/uv/0.8.22/install.sh | sh
   source $HOME/.local/bin/env
   ```

2. Create and activate a virtual environment:

   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. Install NeMo Curator:

   ```bash
   uv pip install torch wheel_stub psutil setuptools setuptools_scm
   echo "transformers==4.55.2" > override.txt
   uv pip install --no-build-isolation "nemo-curator[all]" --override override.txt
   ```

:::

:::{tab-item} Source Installation

Install the latest development version directly from GitHub:

```bash
# Clone the repository
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator

# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with all extras using uv
uv sync --all-extras --all-groups
```

Optional InternVideo2 installation steps:

```bash
bash external/intern_video2_installation.sh
uv add InternVideo/InternVideo2/multi_modality
```

:::

:::{tab-item} Container Installation

NeMo Curator is available as a standalone container on NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator. The container includes NeMo Curator with all dependencies pre-installed. You can run it with:

```bash
# Pull the container from NGC
docker pull nvcr.io/nvidia/nemo-curator:{{ container_version }}

# Run the container with GPU support
docker run --gpus all -it --rm nvcr.io/nvidia/nemo-curator:{{ container_version }}
```

Alternatively, you can build the NeMo Curator container locally using the provided Dockerfile:

```bash
# Build the container locally
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
docker build -t nemo-curator:latest -f docker/Dockerfile .

# Run the container with GPU support
docker run --gpus all -it --rm nemo-curator:latest
```

**Benefits:**

- Pre-configured environment with all dependencies
- Consistent runtime across different systems
- Ideal for production deployments

:::

::::

### Install FFmpeg and Encoders (Required for Video)

Curator’s video pipelines rely on `FFmpeg` for decoding and encoding. If you plan to encode clips (for example, using `--transcode-encoder libopenh264` or `h264_nvenc`), install `FFmpeg` with the corresponding encoders.

::::{tab-set}

:::{tab-item} Debian/Ubuntu (Script)

Use the maintained script in the repository to build and install `FFmpeg` with `libopenh264` and NVIDIA NVENC support. The script enables `--enable-libopenh264`, `--enable-cuda-nvcc`, and `--enable-libnpp`.

- Script source: [docker/common/install_ffmpeg.sh](https://github.com/NVIDIA-NeMo/Curator/blob/main/docker/common/install_ffmpeg.sh)

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/docker/common/install_ffmpeg.sh -o install_ffmpeg.sh
chmod +x install_ffmpeg.sh
sudo bash install_ffmpeg.sh
```

:::

:::{tab-item} Verify Installation

Confirm that `FFmpeg` is on your `PATH` and that at least one H.264 encoder is available:

```bash
ffmpeg -hide_banner -version | head -n 5
ffmpeg -encoders | grep -E "h264_nvenc|libopenh264|libx264" | cat
```

If encoders are missing, reinstall `FFmpeg` with the required options or use the Debian/Ubuntu script above.

:::
::::

### InternVideo2 Support (Optional for Video)

Video processing includes optional support for InternVideo2. To install InternVideo2, run these commands before installing NeMo Curator based on whether you install via PyPI or from source:

::::{tab-set}

:::{tab-item} PyPI Installation
```bash
# Clone and set up InternVideo2
git clone https://github.com/OpenGVLab/InternVideo.git
cd InternVideo
git checkout 09d872e5093296c6f36b8b3a91fc511b76433bf7

# Download and apply NeMo Curator patch
curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Curator/main/external/intern_video2_multimodal.patch -o intern_video2_multimodal.patch
patch -p1 < intern_video2_multimodal.patch
cd ..

# Add InternVideo2 to the environment
uv add InternVideo/InternVideo2/multi_modality
```

:::

:::{tab-item} Source Installation
```bash
# Inside the NeMo Curator folder
bash external/intern_video2_installation.sh
uv add InternVideo/InternVideo2/multi_modality
```

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
* - **text_cpu**
  - `uv pip install nemo-curator[text_cpu]`
  - CPU-only text processing and filtering
* - **text_cuda12**
  - `uv pip install nemo-curator[text_cuda12]`
  - GPU-accelerated text processing with RAPIDS
* - **audio_cpu**
  - `uv pip install nemo-curator[audio_cpu]`
  - CPU-only audio curation with NeMo Toolkit ASR
* - **audio_cuda12**
  - `uv pip install nemo-curator[audio_cuda12]`
  - GPU-accelerated audio curation. When using `uv`, requires `transformers==4.55.2` override.
* - **image_cpu**
  - `uv pip install nemo-curator[image_cpu]`
  - CPU-only image processing
* - **image_cuda12**
  - `uv pip install nemo-curator[image_cuda12]`
  - GPU-accelerated image processing with NVIDIA DALI
* - **video_cpu**
  - `uv pip install nemo-curator[video_cpu]`
  - CPU-only video processing
* - **video_cuda12**
  - `uv pip install --no-build-isolation nemo-curator[video_cuda12]`
  - GPU-accelerated video processing with CUDA libraries. Requires FFmpeg and additional build dependencies when using `uv`.
```

```{note}
**Development Dependencies**: For development tools (pre-commit, ruff, pytest), use `uv sync --group dev --group linting --group test` instead of pip extras. Development dependencies are managed as dependency groups, not optional dependencies.
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
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU detected")
    
    # Check cuDF for GPU deduplication
    import cudf
    print("✓ cuDF available for GPU-accelerated deduplication")
except ImportError as e:
    print(f"⚠ Some GPU modules not available: {e}")
```

### 3. Run a Quickstart Tutorial

Try a modality-specific quickstart to see NeMo Curator in action:

- [Text Curation Quickstart](gs-text) - Set up and run your first text curation pipeline
- [Audio Curation Quickstart](gs-audio) - Get started with audio dataset curation
- [Image Curation Quickstart](gs-image) - Curate image-text datasets for generative models
- [Video Curation Quickstart](gs-video) - Split, encode, and curate video clips at scale
