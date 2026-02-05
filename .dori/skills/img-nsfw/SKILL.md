---
name: img-nsfw
description: Detect and filter NSFW/inappropriate image content. GPU required.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.1"
  modality: image
  gpu-required: "true"
  parent-skill: image
---

# Image NSFW Filtering Skill

Detect and filter inappropriate image content using CLIP-based NSFW classifier. GPU required.

## When This Skill Applies

- User wants to remove NSFW content
- User mentions: "NSFW", "inappropriate", "content moderation", "safe"
- User is building datasets that need to be family-friendly

## NSFW Scoring

The model outputs a probability score (0.0-1.0):
- **< 0.3**: Very likely safe
- **0.3-0.5**: Borderline content
- **> 0.5**: Likely inappropriate

**Important**: Images with scores **below** the threshold are kept (safe content).

## Skill Workflow

### Step 1: Understand the Goal

Ask the user:
1. How strict should filtering be?
2. Is this for public-facing or internal use?
3. What's the acceptable false positive rate?

### Step 2: Generate Pipeline Code

```python
# Image NSFW Filtering Pipeline
# GPU Required: Yes

import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU required for NSFW detection")

print(f"GPU: {torch.cuda.get_device_name(0)}")

from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
from nemo_curator.pipeline import Pipeline

def filter_nsfw(input_path: str, output_path: str, score_threshold: float = 0.5):
    print(f"Input: {input_path}")
    print(f"Threshold: keep images with NSFW score < {score_threshold}")
    
    nsfw_filter = ImageNSFWFilterStage(
        model_dir="/path/to/models",           # Directory to download/load model weights
        score_threshold=score_threshold,        # Keep images with score < threshold
        model_inference_batch_size=32,          # Images per batch (default: 32)
        num_gpus_per_worker=0.25,              # GPU fraction per worker (default: 0.25)
        verbose=False,                         # Enable detailed logging
    )
    
    pipeline = Pipeline(name="nsfw_filter", stages=[nsfw_filter])
    results = pipeline.run()
    
    print(f"Kept {len(results)} safe images")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="/models")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    filter_nsfw(args.input, args.output, args.threshold)
```

## Stage Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dir` | str | None | Directory for model weights |
| `score_threshold` | float | 0.5 | Keep images with score < threshold |
| `model_inference_batch_size` | int | 32 | Images per model inference batch |
| `num_gpus_per_worker` | float | 0.25 | GPU fraction allocated per worker |
| `verbose` | bool | False | Enable detailed logging |

## Threshold Guidelines

| Setting | Threshold | Description |
|---------|-----------|-------------|
| Strict | 0.3 | Block most suggestive content |
| Standard | 0.5 | Block clearly inappropriate (default) |
| Permissive | 0.7 | Block only explicit content |

## Execution

```bash
docker run --gpus all --rm \
    -v $(pwd)/images:/input \
    -v $(pwd)/safe:/output \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/nemo-curator:latest \
    python /data/nsfw_pipeline.py --input /input --output /output --threshold 0.5
```

## Combined with Aesthetic

Often used together for training data:

```python
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage

pipeline = Pipeline(
    name="image_curation",
    stages=[
        ImageEmbeddingStage(model_dir="/models"),           # Generate embeddings first
        ImageNSFWFilterStage(score_threshold=0.5),          # Remove inappropriate (keep < 0.5)
        ImageAestheticFilterStage(score_threshold=0.5),     # Keep quality (keep >= 0.5)
    ],
)
```

## Notes

- Requires CLIP embeddings to be computed first (use `ImageEmbeddingStage`)
- Model weights are automatically downloaded on first run
- Filter logic: **keep images where `nsfw_score < score_threshold`**
