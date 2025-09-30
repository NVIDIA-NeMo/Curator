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

To use NeMo Curator’s video curation modules, ensure you meet the following requirements:

- Python 3.10 or higher
- NVIDIA GPU
  - Volta™ or higher (compute capability 7.0+)
  - CUDA 12 or above
  - With defaults, the full splitting plus captioning example can use up to 38 GB of VRAM. Reduce VRAM to about 21 GB by lowering batch sizes and using FP8 where available.
- `FFmpeg` 7+ on your system path. For H.264, ensure an encoder is available: `h264_nvenc` (GPU) or `libopenh264`/`libx264` (CPU).
- Git (required for some model dependencies)

---

## Install

Create and activate a virtual environment, then choose an install option:

::::{tab-set}

:::{tab-item} GPU (CUDA)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "nemo-curator[video,video_cuda]"
```

:::

:::{tab-item} CPU

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "nemo-curator[video]"
```

:::

::::

## Install `FFmpeg` and Encoders

Curator’s video pipelines rely on `FFmpeg` for decoding and encoding. If you plan to encode clips (for example, using `--transcode-encoder libopenh264` or `h264_nvenc`), install `FFmpeg` with the corresponding encoders.

:::::{tab-set}

::::{tab-item} Debian/Ubuntu (Script)

Use the maintained script in the repository to build and install `FFmpeg` with `libopenh264` and NVIDIA NVENC support. The script enables `--enable-libopenh264`, `--enable-cuda-nvcc`, and `--enable-libnpp`.

- Script source: [docker/common/install_ffmpeg.sh](https://github.com/NVIDIA-NeMo/Curator/blob/main/docker/common/install_ffmpeg.sh)

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/docker/common/install_ffmpeg.sh -o install_ffmpeg.sh
chmod +x install_ffmpeg.sh
sudo bash install_ffmpeg.sh
```

::::

::::{tab-item} macOS (Homebrew)

Install `FFmpeg` using Homebrew, then verify available encoders.

```bash
brew install ffmpeg openh264
```

Note: Homebrew `ffmpeg` does not support NVENC on macOS. If `libopenh264` is not available in your build, use `libx264` as the H.264 encoder instead of `libopenh264`.

::::

::::{tab-item} Verify Installation

Confirm that `FFmpeg` is on your `PATH` and that at least one H.264 encoder is available:

```bash
ffmpeg -hide_banner -version | head -n 5
ffmpeg -encoders | grep -E "h264_nvenc|libopenh264|libx264" | cat
```

If encoders are missing, reinstall `FFmpeg` with the required options or use the Debian/Ubuntu script above.

::::

:::::

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
