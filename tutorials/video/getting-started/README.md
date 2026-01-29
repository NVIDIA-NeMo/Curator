# Getting Started with Video Curation

This tutorial demonstrates how to build video curation pipelines using NeMo Curator's Python API. You will learn to read videos, split them into clips, transcode, filter by quality metrics, generate embeddings and captions, and save results.

**What you will learn**:

- Build a minimal video processing pipeline
- Configure clip splitting algorithms (fixed stride and scene detection)
- Generate embeddings with Cosmos-Embed1 or InternVideo2
- Filter clips by aesthetic quality and motion
- Generate captions with Qwen-VL

**Time**: 15-30 minutes (for a small set of videos)

## Prerequisites

**Hardware**:

- GPU with 16 GB or more of VRAM (required for embedding and captioning stages)
- CPU-only mode works for splitting and transcoding, but GPU stages require a GPU

**Software**:

- Python 3.10 or later
- NeMo Curator installed with video dependencies
- FFmpeg available in the system PATH

**First Run Note**: Model weights (approximately 5 GB total) download automatically on first use. Ensure that you have sufficient disk space and network access.

## Quick Start

Run a minimal pipeline to split videos into fixed-length clips:

```bash
# Set your input and output directories
export VIDEO_DIR="/path/to/your/videos"
export OUTPUT_DIR="/path/to/output"

# Run the pipeline
LOGURU_LEVEL="ERROR" python video_split_clip_example.py \
  --video-dir "$VIDEO_DIR" \
  --output-path "$OUTPUT_DIR" \
  --splitting-algorithm fixed_stride \
  --fixed-stride-split-duration 10.0
```

**What this does**: Processes all videos in `VIDEO_DIR`, splits each into 10-second clips, transcodes them to H.264, and saves results to `OUTPUT_DIR`.

**Expected output**:

```text
Processing 3 videos...
[1/3] video1.mp4: 12 clips extracted
[2/3] video2.mp4: 8 clips extracted
[3/3] video3.mp4: 15 clips extracted
Complete. Results saved to /path/to/output
```

Extend this workflow with embedding, captioning, and filtering options as shown below.

### Output Structure Overview

The pipeline creates the following directory structure:

```text
$OUTPUT_DIR/
├── clips/           # Encoded clip videos (.mp4)
├── metas/v0/        # Per-clip metadata (.json)
├── ce1_embd/        # Cosmos-Embed1 embeddings (.pickle)
└── ce1_embd_parquet/  # Embeddings in Parquet format
```

For a complete description of all output directories, refer to the [Output Structure](#output-structure) section.

### Complete Pipeline Example

This example demonstrates a full video curation pipeline that reads videos, splits into clips, filters by aesthetic quality, generates embeddings, and writes results.

**Key concepts**:

- **Pipeline**: Chains multiple processing stages together
- **XennaExecutor**: The distributed execution engine that runs pipeline stages across available resources (CPU/GPU workers)
- **Stages**: Individual processing steps (reading, splitting, filtering, embedding, writing)

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.video.clipping.clip_extraction_stages import (
    ClipTranscodingStage,
    FixedStrideExtractorStage,
)
from nemo_curator.stages.video.clipping.clip_frame_extraction import ClipFrameExtractionStage
from nemo_curator.stages.video.embedding.cosmos_embed1 import (
    CosmosEmbed1EmbeddingStage,
    CosmosEmbed1FrameCreationStage,
)
from nemo_curator.stages.video.filtering.clip_aesthetic_filter import ClipAestheticFilterStage
from nemo_curator.stages.video.io.clip_writer import ClipWriterStage
from nemo_curator.stages.video.io.video_reader import VideoReader
from nemo_curator.utils.decoder_utils import FrameExtractionPolicy, FramePurpose

# Configuration - update these paths for your environment
VIDEO_DIR = "/path/to/videos"   # Directory containing input video files
OUTPUT_DIR = "/path/to/output"  # Directory for output clips and metadata
MODEL_DIR = "./models"          # Directory for model weights (downloaded automatically)

# Build pipeline
pipeline = Pipeline(name="full_video_curation", description="Complete video curation workflow")

# Stage 1: Read videos from directory
pipeline.add_stage(VideoReader(input_video_path=VIDEO_DIR))

# Stage 2: Split into 10-second clips
pipeline.add_stage(FixedStrideExtractorStage(
    clip_len_s=10.0, clip_stride_s=10.0, min_clip_length_s=2.0, limit_clips=0
))

# Stage 3: Transcode clips to H.264 (CPU-based encoding)
pipeline.add_stage(ClipTranscodingStage(encoder="libx264"))

# Stage 4: Extract frames for embeddings and aesthetics
pipeline.add_stage(
    ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        extract_purposes=[FramePurpose.EMBEDDINGS, FramePurpose.AESTHETICS],
    )
)

# Stage 5: Filter by aesthetic quality (removes clips below threshold)
pipeline.add_stage(
    ClipAestheticFilterStage(model_dir=MODEL_DIR, score_threshold=3.5)
)

# Stage 6-7: Generate Cosmos-Embed1 embeddings (GPU required)
pipeline.add_stage(CosmosEmbed1FrameCreationStage(model_dir=MODEL_DIR, variant="336p"))
pipeline.add_stage(CosmosEmbed1EmbeddingStage(model_dir=MODEL_DIR, variant="336p"))

# Stage 8: Write results to disk
pipeline.add_stage(
    ClipWriterStage(
        output_path=OUTPUT_DIR,
        input_path=VIDEO_DIR,
        upload_clips=True,
        dry_run=False,
        generate_embeddings=True,
        generate_previews=False,
        generate_captions=False,
        embedding_algorithm="cosmos-embed1-336p",
    )
)

# Execute the pipeline
executor = XennaExecutor()
pipeline.run(executor)

print(f"Results saved to: {OUTPUT_DIR}")
```

## Command Line Examples

The `video_split_clip_example.py` script provides a convenient CLI for common workflows:

### Basic Splitting with Embeddings

```bash
python video_split_clip_example.py \
  --video-dir "$VIDEO_DIR" \
  --output-path "$OUTPUT_DIR" \
  --splitting-algorithm fixed_stride \
  --fixed-stride-split-duration 10.0 \
  --embedding-algorithm cosmos-embed1-224p
```

### Scene-Aware Splitting with TransNetV2

```bash
python video_split_clip_example.py \
  --video-dir "$VIDEO_DIR" \
  --output-path "$OUTPUT_DIR" \
  --splitting-algorithm transnetv2 \
  --transnetv2-threshold 0.4 \
  --transnetv2-min-length-s 2.0 \
  --transnetv2-max-length-s 10.0 \
  --embedding-algorithm cosmos-embed1-224p
```

### Full Pipeline with Captions and Filtering

```bash
python video_split_clip_example.py \
  --video-dir "$VIDEO_DIR" \
  --output-path "$OUTPUT_DIR" \
  --splitting-algorithm fixed_stride \
  --fixed-stride-split-duration 10.0 \
  --embedding-algorithm cosmos-embed1-224p \
  --generate-captions \
  --aesthetic-threshold 3.5 \
  --motion-filter enable
```

## Embedding Model Options

### Cosmos-Embed1 (Recommended)

Cosmos-Embed1 is a video embedding model optimized for video understanding tasks. It supports three resolution variants:

| Variant | Resolution | GPU Memory | Speed | Use Case |
|---------|------------|------------|-------|----------|
| **224p** | 224x224 | ~8 GB | Fastest | Large-scale processing |
| **336p** (default) | 336x336 | ~16 GB | Balanced | General use |
| **448p** | 448x448 | ~24 GB | Slowest | Highest quality |

```python
from nemo_curator.stages.video.embedding.cosmos_embed1 import (
    CosmosEmbed1EmbeddingStage,
    CosmosEmbed1FrameCreationStage,
)

# Create frames for embedding
pipeline.add_stage(
    CosmosEmbed1FrameCreationStage(
        model_dir="./models",
        variant="336p",  # Options: "224p", "336p" (default), "448p"
        target_fps=2.0,
    )
)

# Generate embeddings (GPU required)
pipeline.add_stage(
    CosmosEmbed1EmbeddingStage(
        model_dir="./models",
        variant="336p",
        gpu_memory_gb=20,
    )
)
```

Output: 512-dimensional embeddings per clip.

### InternVideo2

Alternative embedding model requiring separate installation. InternVideo2 also produces 512-dimensional embeddings.

**Installation** (from the NeMo-Curator repository root):

```bash
# Run the installation script
bash external/intern_video2_installation.sh

# Add the InternVideo2 dependency (using uv package manager)
uv add InternVideo/InternVideo2/multi_modality

# Or with pip:
pip install InternVideo/InternVideo2/multi_modality
```

**Usage**:

```python
from nemo_curator.stages.video.embedding.internvideo2 import (
    InternVideo2FrameCreationStage,
    InternVideo2EmbeddingStage,
)

pipeline.add_stage(InternVideo2FrameCreationStage(model_dir="./models"))
pipeline.add_stage(InternVideo2EmbeddingStage(model_dir="./models"))
```

Output: 512-dimensional embeddings per clip.

## Captioning

Generate descriptive captions for video clips using Qwen-VL:

```python
from nemo_curator.stages.video.caption.caption_enhancement import CaptionEnhancementStage
from nemo_curator.stages.video.caption.caption_generation import CaptionGenerationStage
from nemo_curator.stages.video.caption.caption_preparation import CaptionPreparationStage

# Prepare frames for captioning
pipeline.add_stage(
    CaptionPreparationStage(
        model_variant="qwen",
        prompt_variant="default",
        sampling_fps=2.0,
        window_size=256,
    )
)

# Generate captions with Qwen-VL
pipeline.add_stage(
    CaptionGenerationStage(
        model_dir="./models",
        model_variant="qwen",
        caption_batch_size=8,
        max_output_tokens=512,
    )
)

# Optional: Enhance captions with LLM
pipeline.add_stage(
    CaptionEnhancementStage(
        model_dir="./models",
        model_variant="qwen",
        model_batch_size=128,
    )
)
```

Enable captioning via CLI with `--generate-captions`. Captions appear in the metadata JSON under the `windows` array.

## Output Structure

The pipeline creates the following directory structure:

```text
$OUTPUT_DIR/
├── clips/                          # Encoded clip videos (.mp4)
├── filtered_clips/                 # Filtered-out clips (.mp4)
├── previews/                       # Preview images (.webp)
├── metas/v0/                       # Per-clip metadata (.json)
├── iv2_embd/                       # InternVideo2 embeddings (.pickle)
├── ce1_embd/                       # Cosmos-Embed1 embeddings (.pickle)
├── iv2_embd_parquet/               # InternVideo2 embeddings (Parquet)
├── ce1_embd_parquet/               # Cosmos-Embed1 embeddings (Parquet)
├── processed_videos/               # Video-level metadata
└── processed_clip_chunks/          # Per-chunk statistics
```

## Metadata Schema

Each clip generates a JSON metadata file in `metas/v0/` with the following structure:

```json
{
  "span_uuid": "d2d0b3d1-...",
  "source_video": "/path/to/source/video.mp4",
  "duration_span": [0.0, 5.0],
  "width_source": 1920,
  "height_source": 1080,
  "framerate_source": 30.0,
  "clip_location": "/outputs/clips/d2/d2d0b3d1-....mp4",
  "motion_score": {
    "global_mean": 0.51,
    "per_patch_min_256": 0.29
  },
  "aesthetic_score": 0.72,
  "windows": [
    {
      "start_frame": 0,
      "end_frame": 30,
      "qwen_caption": "A person walks across a room",
      "qwen_lm_enhanced_caption": "A person briskly crosses a bright modern room"
    }
  ],
  "valid": true
}
```

### Metadata Fields

| Field | Description |
| ----- | ----------- |
| `span_uuid` | Unique identifier for the clip |
| `source_video` | Path to the original video file |
| `duration_span` | Start and end times in seconds `[start, end]` |
| `width_source`, `height_source`, `framerate_source` | Original video properties |
| `clip_location` | Path to the encoded clip file |
| `motion_score` | Motion analysis scores (if motion filtering enabled) |
| `aesthetic_score` | Aesthetic quality score (if aesthetic filtering enabled) |
| `windows` | Caption windows with generated text (if captioning enabled) |
| `valid` | Whether the clip passed all filters |

## Embedding Formats

### Parquet Files

Embeddings are stored in Parquet format with two columns:

- `id`: String UUID for the clip
- `embedding`: List of float values (512 dimensions for both InternVideo2 and Cosmos-Embed1)

### Pickle Files

Individual clip embeddings are also saved as `.pickle` files for direct access.
