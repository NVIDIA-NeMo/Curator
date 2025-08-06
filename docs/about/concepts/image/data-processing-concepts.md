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

- **TimmImageEmbedder:** Uses models from the [timm](https://github.com/huggingface/pytorch-image-models) library (e.g., CLIP, ViT, ResNet) for embedding generation. Supports GPU acceleration, batching, and normalization.
- **Custom Embedders:** You can subclass `ImageEmbedder` to use your own model or data loading logic.
- **Normalization:** Embeddings are typically normalized for compatibility with classifiers and similarity search.
- **Distributed Execution:** Embedding generation can be distributed across multiple GPUs or nodes for scalability.

**Example:**
```python
from nemo_curator.image.embedders import TimmImageEmbedder

embedding_model = TimmImageEmbedder(
    "vit_large_patch14_clip_quickgelu_224.openai",
    pretrained=True,
    batch_size=1024,
    num_threads_per_worker=16,
    normalize_embeddings=True,
)
dataset_with_embeddings = embedding_model(dataset)
```

## Classification

Classifiers score or filter images based on their embeddings.
- **Aesthetic Classifier:** Predicts a score (0–10) for subjective image quality.
- **NSFW Classifier:** Predicts a probability (0–1) that an image contains explicit content.
- **Usage:** Classifiers are lightweight and can be run on the GPU immediately after embedding generation.

**Example:**
```python
from nemo_curator.image.classifiers import AestheticClassifier, NsfwClassifier

aesthetic_classifier = AestheticClassifier()
nsfw_classifier = NsfwClassifier()

dataset_with_aesthetic = aesthetic_classifier(dataset_with_embeddings)
dataset_with_nsfw = nsfw_classifier(dataset_with_embeddings)
```

## Filtering

After classification, you can filter images based on classifier scores or metadata fields.
- Remove images with low aesthetic scores
- Remove or flag images with high NSFW scores
- Filter by metadata (e.g., resolution, aspect ratio)

**Example:**
```python
import dask_cudf
filtered = dataset.metadata[(dataset.metadata['aesthetic_score'] > 0.5) & (dataset.metadata['nsfw_score'] < 0.2)]
```

## Deduplication

Semantic deduplication removes near-duplicate images using embedding similarity and clustering.
- Compute embeddings for all images
- Cluster embeddings (e.g., KMeans)
- Remove or flag duplicates based on similarity thresholds

## Pipeline Flow

A typical image curation pipeline:
1. **Load** the dataset (`ImageTextPairDataset.from_webdataset`)
2. **Generate embeddings** (`TimmImageEmbedder` or custom)
3. **Classify** images (Aesthetic, NSFW)
4. **Filter** images by score or metadata
5. **Deduplicate** (optional)
6. **Export** the curated dataset

This modular approach allows you to customize each step for your workflow. 