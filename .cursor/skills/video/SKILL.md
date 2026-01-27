---
name: video
description: |
  Process video datasets for training data curation. Use when the user wants
  to split videos into clips, generate captions, create embeddings, or filter
  video content. Supports scene detection, motion filtering, and quality scoring.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: video
  gpu-required: true
  nemo-curator-version: ">=0.5.0"
compatibility: Requires Python 3.10+, Linux, GPU with 16-24GB memory
disable-model-invocation: true
---

# Video Processing

Process video datasets for training data curation with scene detection, captioning, and embedding generation.

## Platform Requirements

> **IMPORTANT**: NeMo Curator only runs on **Linux**. macOS and Windows are not supported.

| Platform | Support | Solution |
|----------|---------|----------|
| Linux | Native | Install directly with pip/uv |
| macOS | Not supported | Use Docker |
| Windows | Not supported | Use Docker or WSL2 |

## When to Use

- Splitting videos into clips for training data
- Generating captions for video content
- Creating video embeddings (Cosmos-Embed1, InternVideo2)
- Filtering videos by motion, aesthetics, or quality
- Preparing video datasets for multimodal models

## Quick Start

### 1. List Available Video Stages

```bash
python scripts/list_video_stages.py

# With descriptions
python scripts/list_video_stages.py --verbose

# Filter by category
python scripts/list_video_stages.py --category clipping
```

### 2. Estimate Resources

```bash
python scripts/estimate_video_resources.py \
  --input-path /data/videos \
  --clip-method transnetv2 \
  --caption
```

### 3. Generate Pipeline Config

```bash
python scripts/generate_video_config.py \
  --input-path /data/videos \
  --output-path /data/clips \
  --clip-method transnetv2 \
  --caption \
  --embed \
  --output-file video_pipeline.yaml
```

### 4. Execute Pipeline

```bash
python -m nemo_curator.config.run \
  --config-path=. \
  --config-name=video_pipeline
```

## Video Processing Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                    Video Curation Pipeline                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. READ VIDEOS                                                  │
│     └── VideoReader                                              │
│                                                                   │
│  2. SCENE DETECTION / CLIPPING                                   │
│     ├── VideoFrameExtractionStage (GPU 10GB with pynvc)          │
│     ├── TransNetV2ClipExtractionStage (ML-based, GPU 10GB)       │
│     └── OR FixedStrideExtractorStage (fixed duration, CPU)       │
│                                                                   │
│  3. TRANSCODING                                                  │
│     └── ClipTranscodingStage (H.264 encoding)                    │
│                                                                   │
│  4. FILTERING (optional)                                         │
│     ├── MotionVectorDecodeStage (decode motion data, CPU)        │
│     ├── MotionFilterStage (remove static clips, CPU)             │
│     └── ClipAestheticFilterStage (quality scoring, GPU 4GB)      │
│                                                                   │
│  5. CAPTIONING (optional)                                        │
│     ├── CaptionPreparationStage                                  │
│     ├── CaptionGenerationStage (Qwen VL, GPU)                    │
│     └── CaptionEnhancementStage (optional refinement)            │
│                                                                   │
│  6. EMBEDDING (optional)                                         │
│     ├── ClipFrameExtractionStage (required before embedding)     │
│     ├── CosmosEmbed1FrameCreationStage + EmbeddingStage (20GB)   │
│     └── OR InternVideo2FrameCreationStage + EmbeddingStage       │
│                                                                   │
│  7. WRITE CLIPS                                                  │
│     └── ClipWriterStage                                          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Clipping Methods

### TransNetV2 (ML-based Scene Detection)

Detects scene boundaries using a neural network. Best for:
- Movies, TV shows with scene cuts
- Content with clear visual transitions
- When you need semantically meaningful clips

```bash
python scripts/generate_video_config.py \
  --clip-method transnetv2 \
  --transnetv2-threshold 0.4 \
  --transnetv2-min-length 2.0 \
  --transnetv2-max-length 10.0 \
  ...
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.4 | Scene detection sensitivity (0-1) |
| `min_length_s` | 2.0 | Minimum clip length in seconds |
| `max_length_s` | 10.0 | Maximum clip length in seconds |
| `gpu_memory_gb` | 10.0 | GPU memory requirement |

### Fixed Stride (Duration-based)

Splits videos at fixed intervals. Best for:
- Consistent clip lengths
- Continuous content (lectures, surveillance)
- Lower compute requirements

```bash
python scripts/generate_video_config.py \
  --clip-method fixed_stride \
  --fixed-stride-duration 10.0 \
  ...
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_len_s` | 10.0 | Clip length in seconds |
| `clip_stride_s` | 10.0 | Interval between clip starts |
| `min_clip_length_s` | 2.0 | Minimum clip length |
| `limit_clips` | -1 | Maximum clips per video (-1 = unlimited) |

## Embedding Options

### Cosmos-Embed1 (Recommended)

NVIDIA's video embedding model with multiple resolution variants.

| Variant | Resolution | GPU Memory | Use Case |
|---------|------------|------------|----------|
| `cosmos-embed1-224p` | 224x224 | ~16GB | Fast, lower quality |
| `cosmos-embed1-336p` | 336x336 | ~18GB | Balanced |
| `cosmos-embed1-448p` | 448x448 | ~20GB | Best quality |

### InternVideo2

Open-source alternative with competitive quality.

| Requirement | Value |
|-------------|-------|
| GPU Memory | ~16GB |
| Installation | Requires additional dependencies |

## GPU Requirements

| Stage | GPU Memory | Notes |
|-------|------------|-------|
| VideoReader | 0 | CPU-only |
| VideoFrameExtraction | 10GB | With pynvc decoder (or 0 with ffmpeg_cpu) |
| TransNetV2 | 10GB | Scene detection |
| FixedStride | 0 | CPU-only |
| ClipTranscoding | 0-4GB | Optional GPU acceleration |
| MotionVectorDecode | 0 | CPU-only |
| MotionFilter | 0 | CPU by default (optional GPU) |
| AestheticFilter | 4GB | Quality scoring |
| CaptionGeneration | 1 GPU | Qwen VL model (requires full GPU) |
| CaptionEnhancement | 16GB | Qwen LM model |
| ClipFrameExtraction | 0 | CPU-only (required before embedding) |
| CosmosEmbed1 | 20GB | Default variant |
| InternVideo2 | 16GB | Alternative embedder |

## Pipeline Options

| Option | Description | Default |
|--------|-------------|---------|
| `--clip-method` | `transnetv2` or `fixed_stride` | `transnetv2` |
| `--caption` | Generate video captions | `false` |
| `--embed` | Generate video embeddings | `true` |
| `--embed-model` | Embedding model variant | `cosmos-embed1-224p` |
| `--filter-motion` | Filter static clips | `true` |
| `--aesthetic-threshold` | Minimum aesthetic score | `None` (disabled) |
| `--motion-threshold` | Minimum motion score | `0.00098` |

## Common Workflows

### Basic Clip Extraction

```bash
# Caption and embed are disabled by default (caption) or can be disabled (embed)
python scripts/generate_video_config.py \
  --input-path /data/videos \
  --output-path /data/clips \
  --clip-method fixed_stride \
  --fixed-stride-duration 10.0 \
  --no-embed \
  --output-file basic_clips.yaml
```

### Full Training Data Pipeline

```bash
python scripts/generate_video_config.py \
  --input-path /data/videos \
  --output-path /data/clips \
  --clip-method transnetv2 \
  --caption \
  --embed \
  --embed-model cosmos-embed1-336p \
  --filter-motion \
  --aesthetic-threshold 3.5 \
  --output-file full_pipeline.yaml
```

### Caption-Only Pipeline

```bash
# Enable captions, disable embeddings
python scripts/generate_video_config.py \
  --input-path /data/videos \
  --output-path /data/clips \
  --clip-method transnetv2 \
  --caption \
  --no-embed \
  --output-file caption_pipeline.yaml
```

## References

- [CLIPPING_OPTIONS.md](references/CLIPPING_OPTIONS.md) - Detailed clipping comparison
- [CAPTIONING_GUIDE.md](references/CAPTIONING_GUIDE.md) - Qwen VL captioning parameters

## Related Skills

- `/curate` - Full curation workflow (includes video)
- `/stages` - All available stages
- `/setup` - Environment setup
