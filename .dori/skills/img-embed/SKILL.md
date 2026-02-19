---
name: img-embed
description: Generate CLIP embeddings for images for search, similarity, and deduplication. GPU required.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.1"
  modality: image
  gpu-required: "true"
  parent-skill: image
---

# Image Embedding Skill

Generate CLIP embeddings for images using OpenAI CLIP ViT-L/14. GPU required.

## When This Skill Applies

- User wants image embeddings for search/similarity
- User mentions: "CLIP", "embed", "vector", "search", "dedup"
- User is building image search or deduplication systems

## CLIP Model

Uses `openai/clip-vit-large-patch14` (ViT-L/14):
- **Embedding dimensions**: 768
- **Input resolution**: 224x224 (auto-resized)
- **Quality**: High (best for most use cases)

## Skill Workflow

### Step 1: Understand the Goal

Ask the user:
1. What will embeddings be used for? (search, dedup, clustering)
2. Do you need to remove image data after embedding? (saves memory)
3. What batch size fits your GPU memory?

### Step 2: Generate Pipeline Code

```python
# Image Embedding Pipeline
# GPU Required: Yes

import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU required for image embedding")

print(f"GPU: {torch.cuda.get_device_name(0)}")

from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.pipeline import Pipeline

def generate_embeddings(input_path: str, output_path: str):
    print(f"Input: {input_path}")
    print("Model: openai/clip-vit-large-patch14 (768-dim embeddings)")
    
    embedder = ImageEmbeddingStage(
        model_dir="/path/to/models",           # Directory to download/load model weights
        model_inference_batch_size=32,          # Images per batch (default: 32)
        num_gpus_per_worker=0.25,              # GPU fraction per worker (default: 0.25)
        remove_image_data=False,               # Clear image data after embedding (saves memory)
        verbose=False,                         # Enable detailed logging
    )
    
    pipeline = Pipeline(name="image_embedding", stages=[embedder])
    results = pipeline.run()
    
    print(f"Generated embeddings for {len(results)} images")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="/models")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    generate_embeddings(args.input, args.output)
```

## Stage Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dir` | str | None | Directory for model weights |
| `model_inference_batch_size` | int | 32 | Images per model inference batch |
| `num_gpus_per_worker` | float | 0.25 | GPU fraction allocated per worker |
| `remove_image_data` | bool | False | Clear image data after embedding |
| `verbose` | bool | False | Enable detailed logging |

## Use Cases

### Image Search
Store embeddings in vector DB, query with text or image embeddings.

### Image Deduplication
Compute cosine similarity between embeddings:
- 0.99+: Exact duplicates
- 0.95+: Near-duplicates (crops, resizes)
- 0.90+: Similar images

### Clustering
Use embeddings for K-means or hierarchical clustering.

## Execution

```bash
docker run --gpus all --rm \
    -v $(pwd)/images:/input \
    -v $(pwd)/embeddings:/output \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/nemo-curator:latest \
    python /data/embed_pipeline.py --input /input --output /output
```

## Notes

- Uses OpenAI CLIP ViT-L/14 model (768-dimensional embeddings)
- Model weights are automatically downloaded on first run
- Set `remove_image_data=True` to free memory for downstream stages
- Embeddings are L2-normalized (unit vectors)
