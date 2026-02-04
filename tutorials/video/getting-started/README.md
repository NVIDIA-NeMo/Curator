# Getting Started with Video Curation

The Python script in this directory contains examples for how to run video curation workflows with NeMo Curator. See **`video_split_clip_example.py`** for a complete pipeline that reads videos, splits into clips, transcodes, filters, generates embeddings/captions, and saves the results.

## Quick Start

### Prerequisites

1. **Set up directories**:
   ```bash
   export VIDEO_DIR="/path/to/your/videos"  # Video data to be processed
   export OUTPUT_DIR="/path/to/output"
   export MODEL_DIR="./models"  # Will download models if not exist
   ```

2. **Minimal working example**:
   ```bash
   LOGURU_LEVEL="ERROR" python video_split_clip_example.py \
     --video-dir "$VIDEO_DIR" \
     --output-path "$OUTPUT_DIR" \
     --splitting-algorithm fixed_stride \
     --fixed-stride-split-duration 10.0
   ```
The example above demonstrates how to run a minimal video curation pipeline using NeMo Curator. It processes all videos in the specified `VIDEO_DIR`, splits each video into fixed-length clips (10 seconds each, as set by `--fixed-stride-split-duration 10.0`), and saves the resulting clips to `OUTPUT_DIR`. This is a basic workflow to get started with automated video splitting and curation, and can be extended with additional options for embedding, captioning, filtering, and transcoding as shown in later sections.

**Note: Controlling Verbosity**

The `--verbose` boolean argument allows the user to enable stage-specific verbose logging for all stages of the video curation pipeline, which can be useful for debugging. However, even with the `--verbose` argument disabled, the output verbosity will still contain detailed information about the user environment, resource allocations, and pipeline setup. We recommend using `LOGURU_LEVEL="ERROR" python video_split_clip_example.py ...` to minimize logging even further for the most straightforward experience possible.

### Common Use Cases

**Basic video splitting with embeddings**:
```bash
python video_split_clip_example.py \
  --video-dir "$VIDEO_DIR" \
  --output-path "$OUTPUT_DIR" \
  --splitting-algorithm fixed_stride \
  --fixed-stride-split-duration 10.0 \
  --embedding-algorithm cosmos-embed1-224p
```
This example extends from the above example and adds an additional embedding stages using `cosmos-embed1-224p` model. Use `--model-dir "$MODEL_DIR"` if the model is predownloaded.

**Scene-aware splitting with TransNetV2**:
```bash
python video_split_clip_example.py \
  --video-dir "$VIDEO_DIR" \
  --output-path "$OUTPUT_DIR" \
  --splitting-algorithm transnetv2 \
  --transnetv2-threshold 0.4 \
  --transnetv2-min-length-s 2.0 \
  --transnetv2-max-length-s 10.0 \
  --embedding-algorithm cosmos-embed1-224p \
  --transcode-encoder libopenh264 \
  --verbose
```
This example demonstrates a more advanced workflow than the minimal example by using scene-aware splitting with the TransNetV2 algorithm (which detects scene boundaries instead of fixed intervals), applies the Cosmos-Embed1 embedding model to each clip, transcodes the output using the `libopenh264` encoder, and enables verbose logging for more detailed output.

**Full pipeline with captions and filtering**:
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
This example demonstrates the most comprehensive pipeline among the examples above. In addition to splitting videos and generating embeddings, it also generates captions for each clip and applies filtering based on aesthetic and motion scores. This means that only clips meeting the specified quality thresholds (e.g., `--aesthetic-threshold 3.5` and `--motion-filter enable`) will be kept, and captions will be generated for each valid clip. This workflow is useful for curating high-quality, captioned video datasets with automated quality control.


## Output Structure

The pipeline creates the following directory structure:

```
$OUTPUT_DIR/
├── clips/                          # Encoded clip videos (.mp4)
├── filtered_clips/                 # Filtered-out clips (.mp4)
├── previews/                       # Preview images (.webp)
├── metas/v0/                       # Per-clip metadata (.json)
├── ce1_embd/                       # Cosmos-Embed1 embeddings (.pickle)
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

- **`span_uuid`**: Unique identifier for the clip
- **`source_video`**: Path to the original video file
- **`duration_span`**: Start and end times in seconds `[start, end]`
- **`width_source`**, **`height_source`**, **`framerate_source`**: Original video properties
- **`clip_location`**: Path to the encoded clip file
- **`motion_score`**: Motion analysis scores (if motion filtering enabled)
- **`aesthetic_score`**: Aesthetic quality score (if aesthetic filtering enabled)
- **`windows`**: Caption windows with generated text (if captioning enabled)
- **`valid`**: Whether the clip passed all filters

## Embedding Formats

### Parquet Files
Embeddings are stored in Parquet format with two columns:
- **`id`**: String UUID for the clip
- **`embedding`**: List of float values (768 dimensions for Cosmos-Embed1)

### Pickle Files
Individual clip embeddings are also saved as `.pickle` files for direct access.
