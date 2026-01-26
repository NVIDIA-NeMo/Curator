---
description: "Software component versions and compatibility matrix for NeMo Curator releases"
categories: ["reference"]
tags: ["versions", "dependencies", "compatibility", "release"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused", "devops-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(software-versions)=

# Software Component Versions

This page lists software component versions bundled with each NeMo Curator release. Use this reference to verify compatibility with your environment.

## Core Components

```{list-table}
:header-rows: 1
:widths: 40 30

* - Software Component
  - Version
* - NeMo Curator
  - 1.1.0rc0
* - Python
  - 3.10, 3.11, 3.12
* - PyTorch
  - 2.9.0
* - Transformers
  - ≥4.55.2
* - Ray
  - ≥2.52
* - Cosmos-Xenna
  - 0.1.2
* - Pandas
  - ≥2.1.0
* - PyArrow
  - 22.0.0
```

## GPU-Accelerated Components (CUDA 12)

These components are included when you install with `[cuda12]` or modality-specific CUDA extras.

```{list-table}
:header-rows: 1
:widths: 40 30

* - Software Component
  - Version
* - CUDA
  - 12.8.1
* - cuDF
  - 25.10.0
* - cuML
  - 25.10.0
* - RAPIDS MPF
  - 25.10.0
* - vLLM
  - ≥0.14.0
* - Flash Attention
  - ≤2.8.3
```

## Modality-Specific Components

```{list-table}
:header-rows: 1
:widths: 40 30

* - Software Component
  - Version
* - NeMo Toolkit (Audio)
  - 2.4.0
* - TorchVision (Image/Video)
  - 0.24.0
* - TorchAudio (Audio/Video)
  - 2.9.0
* - NVIDIA DALI (Image)
  - nvidia-dali-cuda120
* - PyNvVideoCodec (Video)
  - 2.0.2
* - CV-CUDA (Video)
  - cvcuda_cu12
* - OpenCV (Video)
  - opencv-python
* - PyAV (Video)
  - 13.1.0
```

## Text Processing Components

```{list-table}
:header-rows: 1
:widths: 40 30

* - Software Component
  - Version
* - fastText
  - 0.9.3
* - Trafilatura
  - 2.0.0
* - Sentence Transformers
  - (latest)
* - Jieba
  - 0.42.1
* - MeCab
  - mecab-python3
```

## Container Environment

```{list-table}
:header-rows: 1
:widths: 40 30

* - Component
  - Version
* - Base Image
  - nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
* - UV Package Manager
  - 0.8.22
* - Container Tag
  - nvcr.io/nvidia/nemo-curator:25.09
```

## Extras

NeMo Curator supports modular installation through pip extras. Choose based on your use case:

```{list-table}
:header-rows: 1
:widths: 20 30 50

* - Extra
  - Command
  - Use Case
* - All Modalities
  - `nemo-curator[all]`
  - Full installation for text, image, audio, and video with GPU support
* - Text (GPU)
  - `nemo-curator[text_cuda12]`
  - Document processing with RAPIDS acceleration, fuzzy deduplication, model inference
* - Image (GPU)
  - `nemo-curator[image_cuda12]`
  - Image curation with DALI, aesthetic/NSFW filtering, deduplication
* - Audio (GPU)
  - `nemo-curator[audio_cuda12]`
  - Speech recognition, WER calculation, audio analysis
* - Video (GPU)
  - `nemo-curator[video_cuda12]`
  - Video clipping, scene detection, GPU encoding/decoding
* - Base GPU
  - `nemo-curator[cuda12]`
  - GPU utilities only (for custom setups)
* - Text (CPU)
  - `nemo-curator[text_cpu]`
  - CPU-only text processing (limited features)
```

All GPU installations require the NVIDIA PyPI index:

```bash
uv pip install --extra-index-url https://pypi.nvidia.com nemo-curator[EXTRA]
```

## Platform Support

```{list-table}
:header-rows: 1
:widths: 25 25 50

* - Platform
  - Architecture
  - Support Level
* - Linux
  - x86_64
  - Full support (GPU and CPU)
* - Linux
  - aarch64
  - CPU support only
* - macOS
  - ARM/Intel
  - CPU support (development only)
* - Windows
  - x86_64
  - Not supported
```

## Security Constraints

NeMo Curator enforces minimum versions for dependencies with known security vulnerabilities:

```{list-table}
:header-rows: 1
:widths: 30 25 45

* - Package
  - Minimum Version
  - CVE Addressed
* - aiohttp
  - ≥3.13.3
  - GHSA-6mq8-rvhq-wgg
* - protobuf
  - ≥4.25.8
  - GHSA-8qvm-5x2c-j2w7
* - xgrammar
  - ≥0.1.21
  - GHSA-5cmr-4px5-23pc
* - starlette
  - ≥0.49.1
  - GHSA-7f5h-v6xp-fcq8
* - Ray
  - ≥2.52
  - GHSA-q279-jhrf-cc6v
* - urllib3
  - ≥2.6.3
  - GHSA-38jv-5279-wg99
```

## Verify Installed Versions

To verify the versions installed in your environment, run the following commands:

```bash
# Check NeMo Curator version
python -c "import nemo_curator; print(nemo_curator.__version__)"

# Check core dependencies
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check GPU dependencies (if installed)
python -c "import cudf; print(f'cuDF: {cudf.__version__}')"
```

:::{note}
Version constraints shown with operators (≥, ≤) indicate minimum or maximum compatible versions.
Exact versions installed can vary based on dependency resolution.
:::

## Related Resources

- {ref}`Release Notes <about-release-notes>`: What's new in this release
- {ref}`Migration Guide <migration-guide>`: Upgrade from previous versions
- [NGC Container Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator): Pre-built containers
- [GitHub Repository](https://github.com/NVIDIA-NeMo/Curator): Source code and issue tracker
- [NeMo Framework 25.09 Versions](https://docs.nvidia.com/nemo/megatron-bridge/latest/releases/software-versions.html): Related framework versions
