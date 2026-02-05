---
name: video
description: Process video data including clip extraction, scene detection, captioning, and embedding generation. Use when the user wants to process videos, extract clips, generate captions, or create video embeddings. GPU required.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.1"
  modality: video
  gpu-required: "true"
---

# Video Curation Skill

Help users process and curate video data using NeMo Curator. This skill covers clipping, captioning, embedding, and filtering of video content.

## When This Skill Applies

- User wants to process video data
- User mentions: "video", "clips", "scenes", "captions", "video embeddings"
- User is building video datasets for training or retrieval

## GPU Requirements Summary

| Stage | GPU Memory | Notes |
|-------|------------|-------|
| VideoReader | CPU | Reads files and metadata |
| VideoFrameExtractionStage | 10GB (pynvc) or CPU | Required before TransNetV2 |
| TransNetV2ClipExtractionStage | 10GB | Scene detection |
| FixedStrideExtractorStage | CPU | Fixed interval clips |
| ClipTranscodingStage | CPU or GPU | Encodes clips to MP4 |
| MotionVectorDecodeStage | CPU | Decodes motion data |
| MotionFilterStage | CPU | Filters by motion |
| ClipAestheticFilterStage | 4GB | Quality scoring |
| CaptionPreparationStage | CPU | Prepares VL inputs |
| CaptionGenerationStage | 1 GPU | Qwen VL captioning |
| CaptionEnhancementStage | 1 GPU | Qwen LM enhancement |
| ClipFrameExtractionStage | CPU | Extract frames for embedding |
| CosmosEmbed1FrameCreationStage | CPU | Prepare Cosmos inputs |
| CosmosEmbed1EmbeddingStage | 20GB | Cosmos embeddings |
| InternVideo2FrameCreationStage | CPU | Prepare IV2 inputs |
| InternVideo2EmbeddingStage | 10GB | InternVideo2 embeddings |

## Skill Workflow

### Step 1: Understand the Goal

Ask the user:

1. What's your end goal? (training data, retrieval, search)
2. What format are your videos? (MP4, directories, URLs)
3. What processing do you need?
   - Clip extraction (split into scenes/fixed intervals)
   - Captioning (generate text descriptions)
   - Embedding (vector representations)
   - Filtering (quality, motion, aesthetics)

### Step 2: Recommend Pipeline Stages

| Goal | Recommended Stages |
|------|-------------------|
| Training dataset | Read → FrameExtract → Clip → Transcode → Caption → Filter → Write |
| Video search/retrieval | Read → FrameExtract → Clip → Transcode → ClipFrameExtract → Embed → Write |
| Quality filtering | Read → FrameExtract → Clip → Transcode → MotionDecode → MotionFilter → AestheticFilter → Write |
| Scene segmentation | Read → FrameExtract → TransNetV2 → Transcode → Write |

### Step 3: Explain Available Stages

**Video Reading**
- `VideoReader`: Reads video files from directory and extracts metadata
- Supports: MP4, AVI, MKV, MOV, WebM

**Clipping**
- `TransNetV2ClipExtractionStage`: AI-based scene detection (10GB GPU)
  - Parameters: `min_length_s`, `max_length_s`, `threshold`
  - Requires `VideoFrameExtractionStage` first
- `FixedStrideExtractorStage`: Fixed-interval clips (CPU only)
  - Parameters: `clip_len_s`, `clip_stride_s`, `min_clip_length_s`

**Captioning** (uses Qwen VL)
- `CaptionPreparationStage`: Prepare inputs (CPU)
- `CaptionGenerationStage`: Generate captions (1 GPU)
  - Parameters: `caption_batch_size`, `prompt_variant`
- `CaptionEnhancementStage`: Enhance captions (1 GPU, optional)

**Embedding**
- Cosmos-Embed1 (NVIDIA):
  - `CosmosEmbed1FrameCreationStage`: Prepare frames (CPU)
  - `CosmosEmbed1EmbeddingStage`: Generate embeddings (20GB GPU)
- InternVideo2:
  - `InternVideo2FrameCreationStage`: Prepare frames (CPU)
  - `InternVideo2EmbeddingStage`: Generate embeddings (10GB GPU)

**Filtering**
- `MotionVectorDecodeStage`: Decode motion data (CPU)
- `MotionFilterStage`: Filter by motion amount (CPU)
  - Parameters: `global_mean_threshold`, `per_patch_min_256_threshold`
- `ClipAestheticFilterStage`: Filter by visual quality (4GB GPU)
  - Parameters: `score_threshold`, `reduction`

### Step 4: Generate Pipeline Code

```python
# Video Processing Pipeline
# GPU Required: Yes

import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU required for video processing")

print(f"GPU: {torch.cuda.get_device_name(0)}")

# Video I/O
from nemo_curator.stages.video.io.video_reader import VideoReader
from nemo_curator.stages.video.io.clip_writer import ClipWriterStage

# Clipping
from nemo_curator.stages.video.clipping.video_frame_extraction import VideoFrameExtractionStage
from nemo_curator.stages.video.clipping.transnetv2_extraction import TransNetV2ClipExtractionStage
from nemo_curator.stages.video.clipping.clip_extraction_stages import ClipTranscodingStage

# Captioning
from nemo_curator.stages.video.caption.caption_preparation import CaptionPreparationStage
from nemo_curator.stages.video.caption.caption_generation import CaptionGenerationStage

# Embedding
from nemo_curator.stages.video.clipping.clip_frame_extraction import ClipFrameExtractionStage
from nemo_curator.stages.video.embedding.cosmos_embed1 import (
    CosmosEmbed1FrameCreationStage,
    CosmosEmbed1EmbeddingStage,
)

from nemo_curator.pipeline import Pipeline

def process_videos(input_dir: str, output_dir: str):
    """Process videos: clip, caption, embed."""
    
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Stage 1: Read videos
    print("\n[1/7] Reading videos...")
    reader = VideoReader(input_video_path=input_dir)
    
    # Stage 2: Extract frames for scene detection (10GB GPU or CPU)
    print("[2/7] Extracting frames...")
    frame_extractor = VideoFrameExtractionStage(
        output_hw=(27, 48),
        decoder_mode="pynvc",  # "pynvc" (GPU) or "ffmpeg_cpu"
    )
    
    # Stage 3: Extract clips using scene detection (10GB GPU)
    print("[3/7] Extracting clips (TransNetV2)...")
    clipper = TransNetV2ClipExtractionStage(
        min_length_s=2.0,      # Minimum 2 seconds
        max_length_s=10.0,     # Maximum 10 seconds
        threshold=0.4,         # Scene detection sensitivity
        gpu_memory_gb=10,
    )
    
    # Stage 4: Transcode clips to standard format
    print("[4/7] Transcoding clips...")
    transcoder = ClipTranscodingStage(
        encoder="libx264",
        encode_batch_size=16,
    )
    
    # Stage 5: Prepare caption inputs (CPU)
    print("[5/7] Preparing captions...")
    caption_prep = CaptionPreparationStage(
        model_variant="qwen",
        prompt_variant="default",
        sampling_fps=2.0,
    )
    
    # Stage 6: Generate captions (1 GPU - Qwen VL)
    print("[6/7] Generating captions...")
    captioner = CaptionGenerationStage(
        model_dir="models/qwen",
        model_variant="qwen",
        caption_batch_size=16,
    )
    
    # Stage 7: Prepare and generate embeddings (20GB GPU - Cosmos-Embed1)
    print("[7/7] Generating embeddings...")
    clip_frame_extractor = ClipFrameExtractionStage(target_fps=2.0)
    embed_frame_creator = CosmosEmbed1FrameCreationStage(variant="336p")
    embedder = CosmosEmbed1EmbeddingStage(variant="336p", gpu_memory_gb=20)
    
    # Build pipeline
    pipeline = Pipeline(
        name="video_curation",
        stages=[
            reader,
            frame_extractor,
            clipper,
            transcoder,
            caption_prep,
            captioner,
            clip_frame_extractor,
            embed_frame_creator,
            embedder,
        ],
    )
    
    # Run
    results = pipeline.run()
    
    print(f"\nDone! Processed {len(results)} videos")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video directory")
    parser.add_argument("--output", required=True, help="Output clip directory")
    args = parser.parse_args()
    process_videos(args.input, args.output)
```

### Step 5: Execution Instructions

**Docker with GPU:**
```bash
docker run --gpus all --rm \
    -v $(pwd)/videos:/input \
    -v $(pwd)/clips:/output \
    nvcr.io/nvidia/nemo-curator:latest \
    python /data/video_pipeline.py --input /input --output /output
```

**Resource requirements:**
- TransNetV2 (scene detection): 10GB GPU
- Captioning (Qwen VL): 1 full GPU
- Cosmos-Embed1: 20GB GPU

### Step 6: Help with Filtering

After clipping, help filter clips:

```python
from nemo_curator.stages.video.filtering.motion_filter import (
    MotionVectorDecodeStage,
    MotionFilterStage,
)
from nemo_curator.stages.video.filtering.clip_aesthetic_filter import ClipAestheticFilterStage

# Decode motion vectors (CPU, required before motion filter)
motion_decoder = MotionVectorDecodeStage(
    target_fps=2.0,
    target_duration_ratio=0.5,
)

# Filter by motion (remove static clips)
motion_filter = MotionFilterStage(
    global_mean_threshold=0.00098,
    per_patch_min_256_threshold=0.000001,
    score_only=False,  # Set True to only compute scores without filtering
)

# Filter by aesthetic quality (4GB GPU)
aesthetic_filter = ClipAestheticFilterStage(
    score_threshold=3.5,
    reduction="min",  # "min" or "mean" across frames
    target_fps=1.0,
)
```

---

## Available Stages (Reference)

### I/O Stages
| Stage | Location | Purpose |
|-------|----------|---------|
| `VideoReader` | `stages/video/io/video_reader.py` | Read video files |
| `ClipWriterStage` | `stages/video/io/clip_writer.py` | Write video clips and metadata |

### Clipping Stages
| Stage | Location | Purpose |
|-------|----------|---------|
| `VideoFrameExtractionStage` | `stages/video/clipping/video_frame_extraction.py` | Extract frames for scene detection |
| `TransNetV2ClipExtractionStage` | `stages/video/clipping/transnetv2_extraction.py` | AI scene detection |
| `FixedStrideExtractorStage` | `stages/video/clipping/clip_extraction_stages.py` | Fixed-interval clips |
| `ClipTranscodingStage` | `stages/video/clipping/clip_extraction_stages.py` | Transcode clips to MP4 |
| `ClipFrameExtractionStage` | `stages/video/clipping/clip_frame_extraction.py` | Extract frames from clips |

### Captioning Stages
| Stage | Location | Purpose |
|-------|----------|---------|
| `CaptionPreparationStage` | `stages/video/caption/caption_preparation.py` | Prepare VL model inputs |
| `CaptionGenerationStage` | `stages/video/caption/caption_generation.py` | Generate captions (Qwen VL) |
| `CaptionEnhancementStage` | `stages/video/caption/caption_enhancement.py` | Enhance captions (Qwen LM) |

### Embedding Stages
| Stage | Location | Purpose |
|-------|----------|---------|
| `CosmosEmbed1FrameCreationStage` | `stages/video/embedding/cosmos_embed1.py` | Prepare Cosmos inputs |
| `CosmosEmbed1EmbeddingStage` | `stages/video/embedding/cosmos_embed1.py` | Cosmos video embeddings |
| `InternVideo2FrameCreationStage` | `stages/video/embedding/internvideo2.py` | Prepare InternVideo2 inputs |
| `InternVideo2EmbeddingStage` | `stages/video/embedding/internvideo2.py` | InternVideo2 embeddings |

### Filtering Stages
| Stage | Location | Purpose |
|-------|----------|---------|
| `MotionVectorDecodeStage` | `stages/video/filtering/motion_filter.py` | Decode motion vectors |
| `MotionFilterStage` | `stages/video/filtering/motion_filter.py` | Filter by motion |
| `ClipAestheticFilterStage` | `stages/video/filtering/clip_aesthetic_filter.py` | Filter by visual quality |

---

## Related Skills

For specialized workflows, see:
- `vid-clip`: Detailed clipping workflows
- `vid-caption`: Detailed captioning workflows
- `vid-embed`: Detailed embedding workflows

---

## Example Interactions

### Example 1: Training dataset
```
User: I want to create a video training dataset from my MP4 files

Agent: I can help build a video curation pipeline. Questions:

1. How many videos, and how long are they on average?
2. Do you want AI-based scene detection or fixed-interval clips?
3. Do you need captions for the clips?
4. What GPU do you have? (Cosmos-Embed1 needs 20GB, Qwen VL needs 1 full GPU)

User: 1000 videos, average 10 minutes. I want scene detection and captions. A100 80GB.

Agent: For scene detection + captioning on A100:
- TransNetV2 for clipping (10GB)
- Qwen VL for captioning (1 GPU)
- You can run both stages sequentially on your A100

Expected output: ~50-100 clips per video, each with a text caption.

Here's your pipeline: [generates code]
```

### Example 2: Video search
```
User: I want to build a video search system

Agent: For video search, you need embeddings for similarity search:

1. Clip extraction (TransNetV2 or fixed stride)
2. Embedding generation (Cosmos-Embed1 recommended, needs 20GB GPU)
3. Store embeddings in vector DB

The pipeline will output clip files + embedding vectors.

Do you have a preferred vector database? (Milvus, Pinecone, FAISS, etc.)
```
