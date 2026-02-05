---
name: vid-clip
description: Extract video clips using scene detection (TransNetV2, 10GB GPU) or fixed intervals (CPU only).
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.1"
  modality: video
  gpu-required: "true"
  parent-skill: video
---

# Video Clipping Skill

Extract video clips using AI scene detection or fixed intervals.

## When This Skill Applies

- User wants to split videos into clips
- User mentions: "clips", "scenes", "extract", "split", "segment"
- User has MP4/video files to process

## Clipping Methods

### TransNetV2 (AI Scene Detection)

- Detects shot boundaries automatically using neural network
- Best for: movies, TV, edited content with scene cuts
- GPU memory: **10GB**
- Parameters: `threshold`, `min_length_s`, `max_length_s`, `max_length_mode`

### Fixed Stride (CPU Only)

- Clips at fixed intervals
- Best for: long-form, unedited content (lectures, surveillance)
- GPU memory: **0GB** (CPU only)
- Parameters: `clip_len_s`, `clip_stride_s`, `min_clip_length_s`

## Full Pipeline Workflow

Video clipping requires multiple stages:

1. **VideoReader** - Read video files and extract metadata
2. **VideoFrameExtractionStage** - Extract frames for scene detection (TransNetV2 only)
3. **TransNetV2ClipExtractionStage** OR **FixedStrideExtractorStage** - Detect/create clip boundaries
4. **ClipTranscodingStage** - Transcode clips to standard format

## Skill Workflow

### Step 1: Understand the Goal

Ask the user:

1. What kind of videos? (movies, vlogs, surveillance, etc.)
2. How long should clips be? (min/max duration)
3. AI scene detection or fixed intervals?

### Step 2: Generate Pipeline Code

#### TransNetV2 (Scene Detection)

```python
# Video Clipping Pipeline with TransNetV2
# GPU Required: Yes (10GB for frame extraction + scene detection)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU required for TransNetV2 scene detection")

print(f"GPU: {torch.cuda.get_device_name(0)}")

from nemo_curator.stages.video.io.video_reader import VideoReader
from nemo_curator.stages.video.clipping.video_frame_extraction import VideoFrameExtractionStage
from nemo_curator.stages.video.clipping.transnetv2_extraction import TransNetV2ClipExtractionStage
from nemo_curator.stages.video.clipping.clip_extraction_stages import ClipTranscodingStage
from nemo_curator.pipeline import Pipeline

def extract_clips_transnetv2(
    input_dir: str,
    output_dir: str,
    min_length_s: float = 2.0,
    max_length_s: float = 10.0,
    threshold: float = 0.4,
):
    """Extract clips using TransNetV2 scene detection.
    
    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save clips
        min_length_s: Minimum clip length in seconds
        max_length_s: Maximum clip length in seconds
        threshold: Scene detection sensitivity (0-1, lower = more sensitive)
    """
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Clip length: {min_length_s}s - {max_length_s}s")
    
    # Stage 1: Read video files
    reader = VideoReader(input_video_path=input_dir)
    
    # Stage 2: Extract frames for scene detection (GPU)
    frame_extractor = VideoFrameExtractionStage(
        output_hw=(27, 48),     # TransNetV2 expects 27x48 frames
        decoder_mode="pynvc",   # Use GPU decoder (falls back to CPU if unavailable)
        verbose=False,
    )
    
    # Stage 3: Detect scene boundaries (GPU)
    clipper = TransNetV2ClipExtractionStage(
        threshold=threshold,           # Scene detection sensitivity
        min_length_s=min_length_s,     # Minimum clip duration
        max_length_s=max_length_s,     # Maximum clip duration
        max_length_mode="stride",      # "stride" or "truncate" for long scenes
        crop_s=0.5,                    # Crop frames from scene boundaries
        gpu_memory_gb=10,              # GPU memory requirement
        verbose=False,
    )
    
    # Stage 4: Transcode clips to standard format
    transcoder = ClipTranscodingStage(
        encoder="libx264",             # "libx264", "libopenh264", or "h264_nvenc"
        encode_batch_size=16,
        verbose=False,
    )
    
    pipeline = Pipeline(
        name="clip_extraction",
        stages=[reader, frame_extractor, clipper, transcoder],
    )
    results = pipeline.run()
    
    print(f"Extracted clips from {len(results)} videos")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video directory")
    parser.add_argument("--output", required=True, help="Output clip directory")
    parser.add_argument("--min-length", type=float, default=2.0)
    parser.add_argument("--max-length", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    extract_clips_transnetv2(args.input, args.output, args.min_length, args.max_length, args.threshold)
```

#### Fixed Stride (CPU Only)

```python
# Video Clipping Pipeline with Fixed Stride
# GPU Required: No (CPU only)

from nemo_curator.stages.video.io.video_reader import VideoReader
from nemo_curator.stages.video.clipping.clip_extraction_stages import (
    FixedStrideExtractorStage,
    ClipTranscodingStage,
)
from nemo_curator.pipeline import Pipeline

def extract_clips_fixed_stride(
    input_dir: str,
    output_dir: str,
    clip_len_s: float = 10.0,
    clip_stride_s: float = 10.0,
    min_clip_length_s: float = 2.0,
    limit_clips: int = -1,
):
    """Extract clips using fixed stride intervals.
    
    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save clips
        clip_len_s: Length of each clip in seconds
        clip_stride_s: Interval between clip starts in seconds
        min_clip_length_s: Minimum clip length (for final clip)
        limit_clips: Maximum clips per video (-1 = unlimited)
    """
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Clip length: {clip_len_s}s, stride: {clip_stride_s}s")
    
    # Stage 1: Read video files
    reader = VideoReader(input_video_path=input_dir)
    
    # Stage 2: Create clip boundaries at fixed intervals (CPU only)
    clipper = FixedStrideExtractorStage(
        clip_len_s=clip_len_s,           # Duration of each clip
        clip_stride_s=clip_stride_s,     # Interval between clip starts
        min_clip_length_s=min_clip_length_s,  # Minimum clip length
        limit_clips=limit_clips,         # Maximum clips per video
        verbose=False,
    )
    
    # Stage 3: Transcode clips to standard format
    transcoder = ClipTranscodingStage(
        encoder="libx264",
        encode_batch_size=16,
        verbose=False,
    )
    
    pipeline = Pipeline(
        name="clip_extraction",
        stages=[reader, clipper, transcoder],
    )
    results = pipeline.run()
    
    print(f"Extracted clips from {len(results)} videos")
    return results
```

## Available Stages

| Stage | Purpose | GPU Memory | Location |
|-------|---------|------------|----------|
| `VideoReader` | Read videos and metadata | CPU | `stages.video.io.video_reader` |
| `VideoFrameExtractionStage` | Extract frames for scene detection | 10GB (pynvc) or CPU | `stages.video.clipping.video_frame_extraction` |
| `TransNetV2ClipExtractionStage` | AI scene detection | 10GB | `stages.video.clipping.transnetv2_extraction` |
| `FixedStrideExtractorStage` | Fixed interval clips | CPU | `stages.video.clipping.clip_extraction_stages` |
| `ClipTranscodingStage` | Transcode clips to MP4 | CPU or GPU | `stages.video.clipping.clip_extraction_stages` |

## TransNetV2 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.4 | Scene detection sensitivity (0-1, lower = more scenes) |
| `min_length_s` | 2.0 | Minimum clip length in seconds |
| `max_length_s` | 10.0 | Maximum clip length in seconds |
| `max_length_mode` | "stride" | How to handle long scenes: "stride" or "truncate" |
| `crop_s` | 0.5 | Seconds to crop from scene boundaries |
| `entire_scene_as_clip` | True | If no transitions found, use entire video as clip |
| `gpu_memory_gb` | 10 | GPU memory requirement |

## FixedStrideExtractor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_len_s` | - | Length of each clip in seconds (required) |
| `clip_stride_s` | - | Interval between clip starts in seconds (required) |
| `min_clip_length_s` | - | Minimum clip length for final clip (required) |
| `limit_clips` | -1 | Maximum clips per video (-1 = unlimited) |

## Transcoding Options

| Encoder | GPU | Description |
|---------|-----|-------------|
| `libx264` | No | Software H.264 encoder (default) |
| `libopenh264` | No | OpenH264 encoder |
| `h264_nvenc` | Yes | NVIDIA GPU encoder (faster) |

## Execution

```bash
# TransNetV2 (requires GPU)
docker run --gpus all --rm \
    -v $(pwd)/videos:/input \
    -v $(pwd)/clips:/output \
    nvcr.io/nvidia/nemo-curator:latest \
    python /data/clip_pipeline.py --input /input --output /output

# Fixed Stride (CPU only)
docker run --rm \
    -v $(pwd)/videos:/input \
    -v $(pwd)/clips:/output \
    nvcr.io/nvidia/nemo-curator:latest \
    python /data/fixed_stride_pipeline.py --input /input --output /output
```

## Choosing a Method

| Use Case | Recommended Method |
|----------|-------------------|
| Movies, TV shows | TransNetV2 (detects scene cuts) |
| Vlogs, YouTube | TransNetV2 |
| Surveillance footage | Fixed Stride |
| Lectures, presentations | Fixed Stride |
| Dashcam video | Either (TransNetV2 for varied content) |
| Continuous recordings | Fixed Stride |
