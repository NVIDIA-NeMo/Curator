---
name: vid-embed
description: Generate video embeddings for search and similarity using Cosmos-Embed1 or InternVideo2. GPU required (20GB for Cosmos, 10GB for InternVideo2).
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.1"
  modality: video
  gpu-required: "true"
  parent-skill: video
---

# Video Embedding Skill

Generate vector embeddings for videos using Cosmos-Embed1 or InternVideo2. GPU required.

## When This Skill Applies

- User wants video embeddings for search/similarity
- User mentions: "embed", "vector", "search", "similarity", "retrieval"
- User is building video search or deduplication systems

## Embedding Models

### Cosmos-Embed1 (NVIDIA)

- Best for general video understanding
- GPU memory: **20GB** (default variant: 336p)
- Variants: `224p` (16GB), `336p` (18GB), `448p` (20GB)
- Output: High-dimensional vectors

### InternVideo2

- Good for action recognition
- GPU memory: **10GB**
- Trained on action datasets

## Important: Two-Stage Embedding Process

Video embedding requires **two stages**:

1. **Frame Creation Stage** - Extracts and prepares frames from video clips
2. **Embedding Stage** - Generates embeddings from prepared frames

Both stages must be included in the pipeline.

## Skill Workflow

### Step 1: Understand the Goal

Ask the user:

1. What will embeddings be used for? (search, dedup, clustering)
2. What vector database? (FAISS, Milvus, Pinecone)
3. Are clips already extracted and transcoded?

### Step 2: Generate Pipeline Code

```python
# Video Embedding Pipeline
# GPU Required: Yes (20GB for Cosmos-Embed1, 10GB for InternVideo2)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU required for video embedding")

print(f"GPU: {torch.cuda.get_device_name(0)}")

from nemo_curator.stages.video.io.video_reader import VideoReader
from nemo_curator.stages.video.clipping.clip_frame_extraction import ClipFrameExtractionStage
from nemo_curator.stages.video.embedding.cosmos_embed1 import (
    CosmosEmbed1FrameCreationStage,
    CosmosEmbed1EmbeddingStage,
)
from nemo_curator.pipeline import Pipeline

def generate_embeddings(input_dir: str, output_path: str, variant: str = "336p"):
    """Generate video embeddings using Cosmos-Embed1.
    
    Args:
        input_dir: Directory containing video clips (must be transcoded MP4s)
        output_path: Path to save embeddings
        variant: Model variant - "224p", "336p", or "448p"
    """
    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")
    print(f"Variant: {variant}")
    
    # Stage 1: Read video files
    reader = VideoReader(input_video_path=input_dir)
    
    # Stage 2: Extract frames from clips (required before embedding)
    frame_extractor = ClipFrameExtractionStage(
        target_fps=2.0,  # Frames per second to extract
        verbose=False,
    )
    
    # Stage 3: Prepare frames for Cosmos-Embed1 model
    frame_creator = CosmosEmbed1FrameCreationStage(
        model_dir="models/cosmos_embed1",
        variant=variant,
        target_fps=2.0,
        verbose=False,
    )
    
    # Stage 4: Generate embeddings (GPU-intensive)
    embedder = CosmosEmbed1EmbeddingStage(
        model_dir="models/cosmos_embed1",
        variant=variant,
        gpu_memory_gb=20,  # Adjust based on variant
        verbose=False,
    )
    
    pipeline = Pipeline(
        name="video_embedding",
        stages=[reader, frame_extractor, frame_creator, embedder],
    )
    results = pipeline.run()
    
    print(f"Generated embeddings for {len(results)} videos")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video directory")
    parser.add_argument("--output", required=True, help="Output path for embeddings")
    parser.add_argument("--variant", default="336p", choices=["224p", "336p", "448p"])
    args = parser.parse_args()
    generate_embeddings(args.input, args.output, args.variant)
```

### InternVideo2 Alternative

```python
from nemo_curator.stages.video.embedding.internvideo2 import (
    InternVideo2FrameCreationStage,
    InternVideo2EmbeddingStage,
)

# Stage 3: Prepare frames for InternVideo2 model
frame_creator = InternVideo2FrameCreationStage(
    model_dir="InternVideo2",
    target_fps=2.0,
    verbose=False,
)

# Stage 4: Generate embeddings
embedder = InternVideo2EmbeddingStage(
    model_dir="InternVideo2",
    gpu_memory_gb=10.0,
    verbose=False,
)
```

## Available Stages

| Stage | Purpose | GPU Memory |
|-------|---------|------------|
| `ClipFrameExtractionStage` | Extract frames from clips | CPU |
| `CosmosEmbed1FrameCreationStage` | Prepare frames for Cosmos model | CPU |
| `CosmosEmbed1EmbeddingStage` | Generate Cosmos embeddings | 16-20GB |
| `InternVideo2FrameCreationStage` | Prepare frames for IV2 model | CPU |
| `InternVideo2EmbeddingStage` | Generate InternVideo2 embeddings | 10GB |

## Cosmos-Embed1 Variants

| Variant | Resolution | GPU Memory | Use Case |
|---------|------------|------------|----------|
| `224p` | 224x224 | ~16GB | Fast, lower quality |
| `336p` | 336x336 | ~18GB | Balanced (default) |
| `448p` | 448x448 | ~20GB | Best quality |

## Storing Embeddings

After generation, embeddings are stored in `clip.cosmos_embed1_embedding` or `clip.intern_video_2_embedding` as numpy arrays.

To store in a vector database:

```python
# Example: FAISS
import faiss
import numpy as np

# Extract embeddings from results
embeddings = []
for task in results:
    for clip in task.data.clips:
        if clip.cosmos_embed1_embedding is not None:
            embeddings.append(clip.cosmos_embed1_embedding)

embeddings = np.vstack(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "video_index.faiss")
```

## Execution

```bash
docker run --gpus all --rm \
    -v $(pwd)/videos:/input \
    -v $(pwd)/embeddings:/output \
    nvcr.io/nvidia/nemo-curator:latest \
    python /data/embed_pipeline.py --input /input --output /output --variant 336p
```

## Prerequisites

Before running embedding, videos must be:

1. **Clipped** - Split into segments using `TransNetV2ClipExtractionStage` or `FixedStrideExtractorStage`
2. **Transcoded** - Encoded to standard format using `ClipTranscodingStage`

See the `vid-clip` skill for clipping workflows.
