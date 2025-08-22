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

This guide shows how to install Curator and run your first video curation pipeline using the Python API.

## Prerequisites

To use NeMo Curator’s video curation modules, ensure you meet the following requirements:

- Python 3.10 or higher
- NVIDIA GPU
  - Volta™ or higher (compute capability 7.0+)
  - CUDA 12 or above
  - With defaults, the full splitting plus captioning example can use up to 38 GB of VRAM. Reduce VRAM to about 21 GB by lowering batch sizes and using FP8 where available.
- FFmpeg on your system path
- Git (required for some model dependencies)

---

## Install

Create and activate a virtual environment, then choose an install option:

::::{tab-set}

:::{tab-item} GPU (CUDA)
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "ray-curator[video,video_cuda]"
```
:::

:::{tab-item} CPU Only
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "ray-curator[video]"
```
:::

::::

## Prepare model weights

InternVideo2 and BERT weights are required for IV2 embeddings. Prepare this directory layout:

```
<MODEL_DIR>/
  OpenGVLab/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt
  google-bert/bert-large-uncased/
    config.json
    tokenizer.json
    ... (standard tokenizer files)
```

Notes:
- Download the IV2 checkpoint from the OpenGVLab page and accept the terms.
- Download the BERT tokenizer files for `google-bert/bert-large-uncased`.
- You can reuse the same `<MODEL_DIR>` across runs.

## Set up data directories

Store input videos locally or on S3-compatible storage.

For local testing, define paths like:

```
DATA_DIR=/path/to/videos
OUT_DIR=/path/to/output_clips
MODEL_DIR=/path/to/models
```

For S3, configure credentials in `~/.aws/credentials` and use `s3://` paths for `--video-dir` and `--output-clip-path`.

## Run the splitting pipeline example

Use the example script to read videos, split into clips, and write outputs. This runs a Ray pipeline with `XennaExecutor` under the hood.

```bash
python -m ray_curator.examples.video.video_split_clip_example \
  --video-dir "$DATA_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-clip-path "$OUT_DIR" \
  --splitting-algorithm fixed_stride \
  --fixed-stride-split-duration 10.0 \
  --embedding-algorithm internvideo2 \
  --transcode-encoder libopenh264 \
  --verbose
```

Common options:

- `--splitting-algorithm`: `fixed_stride` or `transnetv2`
- `--transnetv2-frame-decoder-mode`: `pynvc`, `ffmpeg_gpu`, or `ffmpeg_cpu`
- `--embedding-algorithm`: `cosmos-embed1-224p`, `cosmos-embed1-336p`, `cosmos-embed1-448p`, or `internvideo2`
- `--generate-captions` and `--generate-previews` to enable captioning and preview generation
- `--transcode-use-hwaccel` and `--transcode-encoder h264_nvenc` to use NVENC when available

## Next Steps

Explore the [Video Curation documentation](video-overview). See the splitting pipeline for more parameters and the semantic deduplication guide to remove near-duplicate clips.
