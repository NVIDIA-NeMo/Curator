---
description: "Step-by-step guide to installing Curator and running your first video curation pipeline"
categories: ["getting-started"]
tags: ["video-curation", "installation", "quickstart", "gpu-accelerated", "ray", "python"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "video-only"
---

(gs-video)=

# Get Started with Video Curation

This guide shows how to install Curator and run your first video curation pipeline.

The [example pipeline](#run-the-splitting-pipeline-example) processes a list of videos, splitting each into 10‑second clips using a fixed stride. It then generates clip‑level embeddings for downstream tasks such as duplicate removal and similarity search.

## Prerequisites

To use NeMo Curator's video curation modules, ensure you meet the following requirements:

- **OS**: Ubuntu 24.04/22.04/20.04 (required for GPU-accelerated processing)
- **Python**: 3.10, 3.11, or 3.12
- **uv** (for package management and installation)
- **NVIDIA GPU** (required)
  - Volta™ or higher (compute capability 7.0+)
  - CUDA 12 or above
  - With defaults, the full splitting plus captioning example can use up to 38 GB of VRAM. Reduce VRAM to about 21 GB by lowering batch sizes and using FP8 where available.
- **FFmpeg** 7+ on your system path. For H.264, ensure an encoder is available: `h264_nvenc` (GPU) or `libopenh264`/`libx264` (CPU).
- **Git** (required for some model dependencies)

:::{tip}
If you don't have `uv` installed, refer to the [Installation Guide](../admin/installation.md) for setup instructions, or install it quickly with:

```bash
curl -LsSf https://astral.sh/uv/0.8.22/install.sh | sh
source $HOME/.local/bin/env
```

:::

---

## Install

Create and activate a virtual environment, then choose an install option:

```{note}
Cosmos-Embed1 (the default) is generally better than InternVideo2 for most video embedding tasks. Consider using Cosmos-Embed1 (`cosmos-embed1-224p`) unless you have specific requirements for InternVideo2.
```

::::{tab-set}

:::{tab-item} PyPi Without internvideo2

```bash
uv pip install torch wheel_stub psutil setuptools setuptools_scm
uv pip install --no-build-isolation "nemo-curator[video_cuda12]"
```

:::

:::{tab-item} Source Without internvideo2

```bash
git clone https://github.com/NVIDIA/NeMo-Curator.git
cd NeMo-Curator
uv sync --extra video_cuda12 --all-groups
source .venv/bin/activate
```

:::

:::{tab-item} PyPi With internvideo2

```bash
# Install base dependencies
uv pip install torch wheel_stub psutil setuptools setuptools_scm
uv pip install --no-build-isolation "nemo-curator[video_cuda12]"

# Clone and set up InternVideo2
git clone https://github.com/OpenGVLab/InternVideo.git
cd InternVideo
git checkout 09d872e5093296c6f36b8b3a91fc511b76433bf7

# Download and apply NeMo Curator patch
curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Curator/main/external/intern_video2_multimodal.patch -o intern_video2_multimodal.patch
patch -p1 < intern_video2_multimodal.patch
cd ..

# Add InternVideo2 to the environment
uv pip install InternVideo/InternVideo2/multi_modality
```

:::

:::{tab-item} Source With internvideo2

```bash
git clone https://github.com/NVIDIA/NeMo-Curator.git
cd NeMo-Curator
uv sync --extra video_cuda12 --all-groups
bash external/intern_video2_installation.sh
uv add InternVideo/InternVideo2/multi_modality
source .venv/bin/activate 
```

:::

:::{tab-item} NeMo Curator Container

NeMo Curator is available as a standalone container:

```bash
# Pull the container
docker pull nvcr.io/nvidia/nemo-curator:{{ container_version }}

# Run the container
docker run --gpus all -it --rm nvcr.io/nvidia/nemo-curator:{{ container_version }}
```

```{seealso}
For details on container environments and configurations, see [Container Environments](reference-infrastructure-container-environments-main).
```

:::

::::

## Install FFmpeg and Encoders

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

Refer to [Clip Encoding](video-process-transcoding) to choose encoders and verify NVENC support on your system.

## Choose Embedding Model

Embeddings convert each video clip into a numeric vector that captures visual and semantic content. Curator uses these vectors to:

- Remove near-duplicate clips during duplicate removal
- Enable similarity search and clustering
- Support downstream analysis such as caption verification

You can choose between two embedding models:

- **Cosmos-Embed1 (default)**: Available in three variants—**cosmos-embed1-224p**, **cosmos-embed1-336p**, and **cosmos-embed1-448p**—which differ in input resolution and accuracy/VRAM tradeoff. All variants are automatically downloaded to `MODEL_DIR` on first run.  
  - [cosmos-embed1-224p on Hugging Face](https://huggingface.co/nvidia/Cosmos-Embed1-224p)
  - [cosmos-embed1-336p on Hugging Face](https://huggingface.co/nvidia/Cosmos-Embed1-336p)
  - [cosmos-embed1-448p on Hugging Face](https://huggingface.co/nvidia/Cosmos-Embed1-448p)
- **InternVideo2 (IV2)**: Open model that requires the IV2 checkpoint and BERT model files to be available locally; higher VRAM usage. 
  - [InternVideo Official Github Page](https://github.com/OpenGVLab/InternVideo)

For this quickstart, we're going to set up support for **Cosmos-Embed1-224p**.

### Prepare Model Weights

For most use cases, you only need to create a model directory. The required model files will be downloaded automatically on first run.

1. Create a model directory:
   ```bash
   mkdir -p "$MODEL_DIR"
   ```
   :::{tip}
   You can reuse the same `<MODEL_DIR>` across runs.
   :::

2. No additional setup is required. The model will be downloaded automatically when first used.

## Set Up Data Directories

Store input videos locally or on S3-compatible storage.

- **Local**: Define paths like:

  ```bash
  DATA_DIR=/path/to/videos
  OUT_DIR=/path/to/output_clips
  MODEL_DIR=/path/to/models
  ```

- **S3**: Configure credentials in `~/.aws/credentials` and use `s3://` paths for `--video-dir` and `--output-clip-path`.

## Run the Splitting Pipeline Example

Use the following example script to read videos, split into clips, and write outputs. This runs a Ray pipeline with `XennaExecutor` under the hood.

```bash
python -m nemo_curator.examples.video.video_split_clip_example \
  --video-dir "$DATA_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-clip-path "$OUT_DIR" \
  --splitting-algorithm fixed_stride \
  --fixed-stride-split-duration 10.0 \
  --embedding-algorithm cosmos-embed1-224p \
  --transcode-encoder libopenh264 \
  --verbose
```

### Options

The example script supports the following options:

```{list-table} Common Options
:header-rows: 1

* - Option
  - Values or Description
* - `--splitting-algorithm`
  - `fixed_stride` | `transnetv2`
* - `--transnetv2-frame-decoder-mode`
  - `pynvc` | `ffmpeg_gpu` | `ffmpeg_cpu`
* - `--embedding-algorithm`
  - `cosmos-embed1-224p` | `cosmos-embed1-336p` | `cosmos-embed1-448p` | `internvideo2`
* - `--generate-captions`, `--generate-previews`
  - Enable captioning and preview generation
* - `--transcode-use-hwaccel`, `--transcode-encoder`
  - Use NVENC when available (for example, `h264_nvenc`). Refer to [Clip Encoding](video-process-transcoding) to verify NVENC support and choose encoders.
```

:::{tip}
To use InternVideo2 instead, set `--embedding-algorithm internvideo2`.
:::

## Next Steps

Explore the [Video Curation documentation](video-overview). For encoding guidance, refer to [Clip Encoding](video-process-transcoding).
