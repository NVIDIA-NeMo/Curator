# Image Curation Packages

Reference for image-specific NeMo Curator dependencies.

## Package Extras

| Extra | Use Case |
|-------|----------|
| `image_cpu` | Image processing without GPU acceleration |
| `image_cuda12` | Full image curation with NVIDIA DALI |

## Dependencies

### image_cpu

| Package | Purpose |
|---------|---------|
| `torchvision` | Image transforms, models |

### image_cuda12 (additional)

| Package | Purpose |
|---------|---------|
| `nvidia-dali-cuda120` | GPU image loading |
| `cudf-cu12` | GPU DataFrames |
| `cuml-cu12` | GPU ML (deduplication) |

## Available Stages

| Stage | Purpose | GPU Memory |
|-------|---------|------------|
| `ImageEmbeddingStage` | CLIP embeddings | ~4 GB (0.25 GPU) |
| `ImageAestheticFilterStage` | Quality scoring | ~4 GB |
| `ImageNSFWFilterStage` | Content safety | ~4 GB |
| `ImageDuplicatesRemovalStage` | Deduplication | Varies |

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| JPEG | `.jpg`, `.jpeg` | Most common |
| PNG | `.png` | Lossless |
| WebP | `.webp` | Efficient |
| GIF | `.gif` | First frame only |
| BMP | `.bmp` | Uncompressed |
| TIFF | `.tiff` | High quality |

## GPU Requirements

Image processing is relatively lightweight:

**Minimum**: 8 GB VRAM for all image stages
**Recommended**: 16 GB for comfortable headroom

## Common Workflows

### CLIP Embedding Generation

```yaml
stages:
  - _target_: nemo_curator.stages.image.embedders.clip_embedder.ImageEmbeddingStage
    model_name: "openai/clip-vit-large-patch14"
    batch_size: 32
```

### Aesthetic Filtering

```yaml
stages:
  - _target_: nemo_curator.stages.image.embedders.clip_embedder.ImageEmbeddingStage
  - _target_: nemo_curator.stages.image.filters.aesthetic_filter.ImageAestheticFilterStage
    min_aesthetic_score: 0.5
```

### NSFW Filtering

```yaml
stages:
  - _target_: nemo_curator.stages.image.filters.nsfw_filter.ImageNSFWFilterStage
    threshold: 0.5
```

## Common Issues

### DALI Installation

DALI requires CUDA 12:

```bash
nvidia-smi  # Check CUDA version
uv pip install nvidia-dali-cuda120
```

### torchvision Compatibility

Ensure torch and torchvision versions match:

```bash
python -c "import torch; import torchvision; print(torch.__version__, torchvision.__version__)"
uv pip install torch torchvision --force-reinstall
```

### Out of Memory

Reduce batch size:

```yaml
- _target_: nemo_curator.stages.image.embedders.clip_embedder.ImageEmbeddingStage
  batch_size: 16  # Reduce from 32
```
