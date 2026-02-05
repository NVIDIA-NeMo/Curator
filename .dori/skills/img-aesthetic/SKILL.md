---
name: img-aesthetic
description: Score and filter images by aesthetic quality using LAION aesthetic predictor. GPU required.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.1"
  modality: image
  gpu-required: "true"
  parent-skill: image
---

# Image Aesthetic Filtering Skill

Score and filter images by visual quality using the LAION aesthetic predictor (`ttj/sac-logos-ava1-l14-linearMSE`). GPU required.

## When This Skill Applies

- User wants to filter by image quality
- User mentions: "aesthetic", "quality", "beautiful", "good looking"
- User is building training datasets that need high-quality images

## Aesthetic Scores

The model outputs normalized scores (0.0-1.0). Higher scores indicate better aesthetic quality.

**Typical thresholds:**
- Permissive: >= 0.3
- Standard: >= 0.5 (default)
- High quality: >= 0.6
- Premium: >= 0.7

## Skill Workflow

### Step 1: Understand the Goal

Ask the user:
1. What quality level do you need?
2. What's the use case? (training, display, filtering)
3. How strict should filtering be?

### Step 2: Generate Pipeline Code

```python
# Image Aesthetic Filtering Pipeline
# GPU Required: Yes

import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU required for aesthetic scoring")

print(f"GPU: {torch.cuda.get_device_name(0)}")

from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.pipeline import Pipeline

def filter_by_aesthetic(input_path: str, output_path: str, score_threshold: float = 0.5):
    print(f"Input: {input_path}")
    print(f"Threshold: >= {score_threshold}")
    
    aesthetic_filter = ImageAestheticFilterStage(
        model_dir="/path/to/models",          # Directory to download/load model weights
        score_threshold=score_threshold,       # Keep images with score >= threshold
        model_inference_batch_size=32,         # Images per batch (default: 32)
        num_gpus_per_worker=0.25,              # GPU fraction per worker (default: 0.25)
        verbose=False,                         # Enable detailed logging
    )
    
    pipeline = Pipeline(name="aesthetic_filter", stages=[aesthetic_filter])
    results = pipeline.run()
    
    print(f"Kept {len(results)} images with aesthetic score >= {score_threshold}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="/models")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    filter_by_aesthetic(args.input, args.output, args.threshold)
```

## Stage Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dir` | str | None | Directory for model weights |
| `score_threshold` | float | 0.5 | Keep images with score >= threshold |
| `model_inference_batch_size` | int | 32 | Images per model inference batch |
| `num_gpus_per_worker` | float | 0.25 | GPU fraction allocated per worker |
| `verbose` | bool | False | Enable detailed logging |

## Threshold Guidelines

| Use Case | Threshold | Expected Keep Rate |
|----------|-----------|-------------------|
| Permissive | 0.3 | ~80% |
| Standard | 0.5 | ~50-60% |
| High quality | 0.6 | ~20-30% |
| Premium | 0.7 | ~5-10% |

## Execution

```bash
docker run --gpus all --rm \
    -v $(pwd)/images:/input \
    -v $(pwd)/filtered:/output \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/nemo-curator:latest \
    python /data/aesthetic_pipeline.py --input /input --output /output --threshold 0.5
```

## Tuning

If too aggressive: lower threshold
If too permissive: raise threshold

Check score distribution first:
```python
# Score all images without filtering, then analyze distribution
scores = [img.aesthetic_score for img in results]
print(f"Min: {min(scores):.3f}, Max: {max(scores):.3f}, Mean: {sum(scores)/len(scores):.3f}")
```

## Notes

- Requires CLIP embeddings to be computed first (use `ImageEmbeddingStage`)
- Model weights are automatically downloaded on first run
- Uses LAION aesthetic predictor: `ttj/sac-logos-ava1-l14-linearMSE`
