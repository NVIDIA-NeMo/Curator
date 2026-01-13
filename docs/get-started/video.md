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

## Overview

This quickstart guide demonstrates how to:

1. **Install NeMo Curator** with video processing support
2. **Set up FFmpeg** with GPU-accelerated encoding
3. **Configure embedding models** (Cosmos-Embed1 or InternVideo2)
4. **Process videos** through a complete splitting and embedding pipeline
5. **Generate outputs** ready for duplicate removal, captioning, and model training

**What you'll build:** A video processing pipeline that:
- Splits videos into 10-second clips using fixed stride or scene detection
- Generates clip-level embeddings for similarity search and deduplication
- Optionally creates captions and preview images
- Outputs results in formats compatible with multimodal training workflows

## Prerequisites

### System Requirements

To use NeMo Curator's video curation capabilities, ensure your system meets these requirements:

#### Operating System
* **Ubuntu 24.04, 22.04, or 20.04** (required for GPU-accelerated video processing)
* Other Linux distributions may work but are not officially supported

#### Python Environment
* **Python 3.10, 3.11, or 3.12**
* **uv package manager** for dependency management
* **Git** for model and repository dependencies

#### GPU Requirements
* **NVIDIA GPU required** (CPU-only mode not supported for video processing)
* **Architecture**: Volta™ or newer (compute capability 7.0+)
  - Examples: V100, T4, RTX 2080+, A100, H100
* **CUDA**: Version 12.0 or above
* **VRAM**: Minimum requirements by configuration:
  - Basic splitting + embedding: ~16GB VRAM
  - Full pipeline (splitting + embedding + captioning): ~38GB VRAM
  - Reduced configuration (lower batch sizes, FP8): ~21GB VRAM

#### Software Dependencies
* **FFmpeg 8.0+** with H.264 encoding support
  - GPU encoder: `h264_nvenc` (recommended for performance)
  - CPU encoders: `libopenh264` or `libx264` (fallback options)

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
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
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
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
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

### Available Models


Embeddings convert each video clip into a numeric vector that captures visual and semantic content. Curator uses these vectors to:

- Remove near-duplicate clips during duplicate removal
- Enable similarity search and clustering
- Support downstream analysis such as caption verification

NeMo Curator supports two embedding model families:

#### Cosmos-Embed1 (Recommended)

**Cosmos-Embed1 (default)**: Available in three variants—**cosmos-embed1-224p**, **cosmos-embed1-336p**, and **cosmos-embed1-448p**—which differ in input resolution and accuracy/VRAM tradeoff. All variants are automatically downloaded to `MODEL_DIR` on first run.

| Model Variant | Resolution | VRAM Usage | Speed | Accuracy | Best For |
|---------------|------------|------------|-------|----------|----------|
| **cosmos-embed1-224p** | 224×224 | ~8GB | Fastest | Good | Large-scale processing, initial curation |
| **cosmos-embed1-336p** | 336×336 | ~12GB | Medium | Better | Balanced performance and quality |
| **cosmos-embed1-448p** | 448×448 | ~16GB | Slower | Best | High-quality embeddings, fine-grained matching |

**Model links:**
- [cosmos-embed1-224p on Hugging Face](https://huggingface.co/nvidia/cosmos-embed1-224p)
- [cosmos-embed1-336p on Hugging Face](https://huggingface.co/nvidia/cosmos-embed1-336p)
- [cosmos-embed1-448p on Hugging Face](https://huggingface.co/nvidia/cosmos-embed1-448p)

#### InternVideo2 (IV2)

Open model that requires the IV2 checkpoint and BERT model files to be available locally; higher VRAM usage. 
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

Organize input videos and output locations before running the pipeline.

- **Local**: For local file processing. Define paths like:

  ```bash
  DATA_DIR=/path/to/videos
  OUT_DIR=/path/to/output_clips
  MODEL_DIR=/path/to/models
  ```

- **S3**: For cloud storage (AWS S3, MinIO, etc.). Configure credentials in `~/.aws/credentials` and use `s3://` paths for `--video-dir` and `--output-clip-path`.

**S3 usage notes:**
- Input videos can be read from S3 paths
- Output clips can be written to S3 paths
- Model directory should remain local for performance
- Ensure IAM permissions allow read/write access to specified buckets

## Run the Splitting Pipeline Example

Use the example script from https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/video/getting-started to read videos, split into clips, and write outputs. This runs a Ray pipeline with `XennaExecutor` under the hood.

```bash
python tutorials/video/getting-started/video_split_clip_example.py \
  --video-dir "$DATA_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-clip-path "$OUT_DIR" \
  --splitting-algorithm fixed_stride \
  --fixed-stride-split-duration 10.0 \
  --embedding-algorithm cosmos-embed1-224p \
  --transcode-encoder libopenh264 \
  --verbose
```

**What this command does:**
1. Reads all video files from `$DATA_DIR`
2. Splits each video into 10-second clips using fixed stride
3. Generates embeddings using Cosmos-Embed1-224p model
4. Encodes clips using libopenh264 codec
5. Writes output clips and metadata to `$OUT_DIR`

### Configuration Options Reference

| Option | Values | Description |
|--------|--------|-------------|
| **Splitting** |
| `--splitting-algorithm` | `fixed_stride`, `transnetv2` | Method for dividing videos into clips |
| `--fixed-stride-split-duration` | Float (seconds) | Clip length for fixed stride (default: 10.0) |
| `--transnetv2-frame-decoder-mode` | `pynvc`, `ffmpeg_gpu`, `ffmpeg_cpu` | Frame decoding method for TransNetV2 |
| **Embedding** |
| `--embedding-algorithm` | `cosmos-embed1-224p`, `cosmos-embed1-336p`, `cosmos-embed1-448p`, `internvideo2` | Embedding model to use |
| **Encoding** |
| `--transcode-encoder` | `h264_nvenc`, `libopenh264`, `libx264` | Video encoder for output clips |
| `--transcode-use-hwaccel` | Flag | Enable hardware acceleration for encoding |
| **Optional Features** |
| `--generate-captions` | Flag | Generate text captions for each clip |
| `--generate-previews` | Flag | Create preview images for each clip |
| `--verbose` | Flag | Enable detailed logging output |

:::{tip}
To use InternVideo2 instead, set `--embedding-algorithm internvideo2`.
:::

### Understanding Pipeline Output

After successful execution, the output directory will contain:

```
$OUT_DIR/
├── clips/
│   ├── video1_clip_0000.mp4
│   ├── video1_clip_0001.mp4
│   └── ...
├── embeddings/
│   ├── video1_clip_0000.npy
│   ├── video1_clip_0001.npy
│   └── ...
├── metadata/
│   └── manifest.jsonl
└── previews/  (if --generate-previews enabled)
    ├── video1_clip_0000.jpg
    └── ...
```

**File descriptions:**
- **clips/**: Encoded video clips (MP4 format)
- **embeddings/**: Numpy arrays containing clip embeddings (for similarity search)
- **metadata/manifest.jsonl**: JSONL file with clip metadata (paths, timestamps, embeddings)
- **previews/**: Thumbnail images for each clip (optional)

**Example manifest entry:**
```json
{
  "video_path": "/data/input_videos/video1.mp4",
  "clip_path": "/data/output_clips/clips/video1_clip_0000.mp4",
  "start_time": 0.0,
  "end_time": 10.0,
  "embedding_path": "/data/output_clips/embeddings/video1_clip_0000.npy",
  "preview_path": "/data/output_clips/previews/video1_clip_0000.jpg"
}
```

## Best Practices

### Data Preparation
- **Validate input videos**: Ensure videos are not corrupted before processing
- **Consistent formats**: Convert videos to a standard format (MP4 with H.264) for consistent results
- **Organize by content**: Group similar videos together for efficient processing

### Model Selection
- **Start with Cosmos-Embed1-224p**: Best balance of speed and quality for initial experiments
- **Upgrade resolution as needed**: Use 336p or 448p only when higher precision is required
- **Monitor VRAM usage**: Check GPU memory with `nvidia-smi` during processing

### Pipeline Configuration
- **Enable verbose logging**: Use `--verbose` flag for debugging and monitoring
- **Test on small subset**: Run pipeline on 5-10 videos before processing large datasets
- **Use GPU encoding**: Enable NVENC for significant performance improvements
- **Save intermediate results**: Keep embeddings and metadata for downstream tasks

### Infrastructure
- **Use shared storage**: Mount shared filesystem for multi-node processing
- **Allocate sufficient VRAM**: Plan for peak usage (captioning + embedding)
- **Monitor GPU utilization**: Use `nvidia-smi dmon` to track GPU usage during processing
- **Schedule long-running jobs**: Process large video datasets in batch jobs overnight

## Next Steps

Explore the [Video Curation documentation](video-overview). For encoding guidance, refer to [Clip Encoding](video-process-transcoding).
