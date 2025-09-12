---
description: "Core concepts for processing image data including embedding generation, classification, filtering, and deduplication"
categories: ["concepts-architecture"]
tags: ["data-processing", "embedding", "classification", "filtering", "deduplication", "gpu-accelerated", "pipeline"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "image-only"
---

(about-concepts-image-data-processing)=
# Data Processing Concepts (Image)

This page covers the core concepts for processing image data in NeMo Curator.

## Embedding Generation

Image embeddings are vector representations of images, used for downstream tasks like classification, filtering, and deduplication.

- **ImageEmbeddingStage:** Uses CLIP ViT-L/14 model for high-quality embedding generation. Supports GPU acceleration, batching, and automatic CPU fallback.
- **Custom Embedding Stages:** You can subclass `ProcessingStage` to use your own model or preprocessing logic.
- **CLIP Integration:** Built-in CLIP model provides robust embeddings for aesthetic and NSFW classification.
- **Pipeline Integration:** Embedding generation integrates seamlessly into NeMo Curator's pipeline architecture.

**Example:**
```python
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage

# Add to pipeline
pipeline.add_stage(ImageEmbeddingStage(
    model_dir="/path/to/models",
    model_inference_batch_size=32,
    num_gpus_per_worker=0.25,
    remove_image_data=False,
))
```

## Classification

Classification stages score and filter images based on their embeddings.

- **ImageAestheticFilterStage:** Predicts aesthetic scores (0–1) and filters images below a threshold.
- **ImageNSFWFilterStage:** Predicts NSFW probability (0–1) and filters images above a threshold.
- **Pipeline Integration:** Classification stages run efficiently after embedding generation in the same pipeline.

**Example:**
```python
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage

# Add to pipeline
pipeline.add_stage(ImageAestheticFilterStage(
    model_dir="/path/to/models",
    score_threshold=0.5,
    model_inference_batch_size=32,
))

pipeline.add_stage(ImageNSFWFilterStage(
    model_dir="/path/to/models", 
    score_threshold=0.5,
    model_inference_batch_size=32,
))
```

## Filtering

Filtering is built into the classification stages, which automatically remove images that don't meet the configured thresholds.
- **Aesthetic Filtering:** Images below `score_threshold` are automatically filtered out
- **NSFW Filtering:** Images above `score_threshold` are automatically filtered out  
- **Pipeline Flow:** Filtering happens seamlessly as part of the stage processing

The filtering is handled automatically by the stages - images that don't meet criteria are removed from the `ImageBatch` before passing to the next stage.

## Deduplication

Semantic deduplication removes near-duplicate images using embedding similarity and clustering.
- Compute embeddings for all images
- Cluster embeddings (e.g., KMeans)
- Remove or flag duplicates based on similarity thresholds

## Pipeline Flow

A typical image curation pipeline using NeMo Curator's stage-based architecture:
1. **Partition** tar files (`FilePartitioningStage`)
2. **Load** images from WebDataset (`ImageReaderStage`)
3. **Generate embeddings** (`ImageEmbeddingStage`)
4. **Filter by aesthetics** (`ImageAestheticFilterStage`)
5. **Filter NSFW content** (`ImageNSFWFilterStage`)
6. **Export** results (`ImageWriterStage`)

This modular pipeline approach allows you to customize, reorder, or skip stages based on your workflow needs. 