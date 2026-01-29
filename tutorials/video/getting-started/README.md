# Getting Started with Video Curation

This tutorial demonstrates the usage of NeMo Curator's Python API to curate video data. We show how to build pipelines that read videos, split them into clips, transcode, filter by quality metrics, generate embeddings and captions, and save results.

The directory contains:
- `video_split_clip_example.py`: A complete CLI-based pipeline with many configurable options

Note: Run these examples on GPUs for best performance.

## Quick Start

### Prerequisites

1. **Set up directories**:

2. **Minimal working example**:
   ```bash
   LOGURU_LEVEL="ERROR" python video_split_clip_example.py \
     --video-dir "$VIDEO_DIR" \
     --output-path "$OUTPUT_DIR" \
     --splitting-algorithm fixed_stride \
     --fixed-stride-split-duration 10.0
   ```
The example above demonstrates how to run a minimal video curation pipeline using NeMo Curator. It processes all videos in the specified `VIDEO_DIR`, splits each video into fixed-length clips (10 seconds each, as set by `--fixed-stride-split-duration 10.0`), and saves the resulting clips to `OUTPUT_DIR`. This is a basic workflow to get started with automated video splitting and curation, and can be extended with additional options for embedding, captioning, filtering, and transcoding as shown in later sections.

```python
from nemo_curator.stages.video.caption.caption_preparation import CaptionPreparationStage
from nemo_curator.stages.video.caption.caption_generation import CaptionGenerationStage
from nemo_curator.stages.video.caption.caption_enhancement import CaptionEnhancementStage

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

### Complete Pipeline Example

Here's a full pipeline combining multiple capabilities:

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.video.io.video_reader import VideoReader
from nemo_curator.stages.video.clipping.clip_extraction_stages import (
    FixedStrideExtractorStage,
    ClipTranscodingStage,
)
from nemo_curator.stages.video.clipping.clip_frame_extraction import ClipFrameExtractionStage
from nemo_curator.stages.video.embedding.cosmos_embed1 import (
    CosmosEmbed1FrameCreationStage,
    CosmosEmbed1EmbeddingStage,
)
from nemo_curator.stages.video.filtering.clip_aesthetic_filter import ClipAestheticFilterStage
from nemo_curator.stages.video.io.clip_writer import ClipWriterStage
from nemo_curator.utils.decoder_utils import FrameExtractionPolicy, FramePurpose

# Configuration
VIDEO_DIR = "/path/to/videos"
OUTPUT_DIR = "/path/to/output"
MODEL_DIR = "./models"

# Build pipeline
pipeline = Pipeline(name="full_video_curation", description="Complete video curation workflow")

# Read → Split → Transcode
pipeline.add_stage(VideoReader(input_video_path=VIDEO_DIR))
pipeline.add_stage(FixedStrideExtractorStage(
    clip_len_s=10.0, clip_stride_s=10.0, min_clip_length_s=2.0, limit_clips=0
))
pipeline.add_stage(ClipTranscodingStage(encoder="libopenh264"))

# Extract frames for embeddings and aesthetics
pipeline.add_stage(
    ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        extract_purposes=[FramePurpose.EMBEDDINGS, FramePurpose.AESTHETICS],
    )
)

# Filter by aesthetic quality
pipeline.add_stage(
    ClipAestheticFilterStage(model_dir=MODEL_DIR, score_threshold=3.5)
)

# Generate embeddings
pipeline.add_stage(CosmosEmbed1FrameCreationStage(model_dir=MODEL_DIR, variant="224p"))
pipeline.add_stage(CosmosEmbed1EmbeddingStage(model_dir=MODEL_DIR, variant="224p"))

# Write results
pipeline.add_stage(
    ClipWriterStage(
        output_path=OUTPUT_DIR,
        input_path=VIDEO_DIR,
        upload_clips=True,
        dry_run=False,
        generate_embeddings=True,
        generate_previews=False,
        generate_captions=False,
        embedding_algorithm="cosmos-embed1-224p",
    )
)

# Execute
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



### InternVideo2

Alternative embedding model requiring separate installation:

```bash
# Install InternVideo2
cd /path/to/Curator
bash external/intern_video2_installation.sh
uv add InternVideo/InternVideo2/multi_modality
```

```python
from nemo_curator.stages.video.embedding.internvideo2 import (
    InternVideo2FrameCreationStage,
    InternVideo2EmbeddingStage,
)

pipeline.add_stage(InternVideo2FrameCreationStage(model_dir="./models"))
pipeline.add_stage(InternVideo2EmbeddingStage(model_dir="./models"))
```

- Output: 512-dimensional embeddings

## Output Structure

The pipeline creates the following directory structure:

```
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
|-------|-------------|
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
- `embedding`: List of float values (512 dims for InternVideo2, 768 for Cosmos-Embed1)

### Pickle Files

Individual clip embeddings are also saved as `.pickle` files for direct access.
