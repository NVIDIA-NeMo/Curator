---
description: "Process image data using embeddings, filters, and filtering for high-quality dataset curation"
categories: ["workflows"]
tags: ["data-processing", "embedding", "filtering", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "image-only"
---

(image-process-data)=

# Process Data for Image Curation

Process image data you've loaded from tar archives using NeMo Curator's suite of tools. These tools help you generate embeddings, filter images, and prepare your dataset to produce high-quality data for downstream AI tasks such as generative model training, dataset analysis, or quality control.

## How it Works

Image processing in NeMo Curator follows a pipeline-based approach with these stages:

1. **Partition files** using `FilePartitioningStage` to distribute tar files
2. **Read images** using `ImageReaderStage` with DALI acceleration
3. **Generate embeddings** using `ImageEmbeddingStage` with CLIP models
4. **Apply filters** using `ImageAestheticFilterStage` and `ImageNSFWFilterStage`
5. **Save results** using `ImageWriterStage` to export curated datasets

Each stage processes `ImageBatch` objects containing images, metadata, and processing results. You can use built-in stages or create custom stages for advanced use cases.

---

## Filter Options

:::: {grid} 1 2 2 2
:gutter: 1 1 1 2

::: {grid-item-card} Aesthetic Filter Stage
:link: image-process-data-filters-aesthetic
:link-type: ref

Assess the subjective quality of images using a model trained on human aesthetic preferences. Filters images based on aesthetic score thresholds.
+++
{bdg-secondary}`ImageAestheticFilterStage` {bdg-secondary}`aesthetic_score`
:::

::: {grid-item-card} NSFW Filter Stage
:link: image-process-data-filters-nsfw
:link-type: ref

Detect not-safe-for-work (NSFW) content in images using a CLIP-based filter. Filters explicit material from your datasets.
+++
{bdg-secondary}`ImageNSFWFilterStage` {bdg-secondary}`nsfw_score`
:::

::::

## Embedding Options

:::: {grid} 1 2 2 2
:gutter: 1 1 1 2

::: {grid-item-card} CLIP Embedding Stage
:link: image-process-data-embeddings-clip
:link-type: ref

Generate image embeddings using CLIP models with GPU acceleration. Supports various CLIP architectures and automatic model downloading.
+++
{bdg-secondary}`ImageEmbeddingStage` {bdg-secondary}`CLIP` {bdg-secondary}`GPU-accelerated`
:::


::::

## Filtering Images

The filtering stages (`ImageAestheticFilterStage`, `ImageNSFWFilterStage`) include filtering capabilities. Images that don't meet the specified thresholds are automatically filtered out during processing.

**Built-in filtering capabilities:**

* **Aesthetic filtering**: Remove images with low aesthetic scores using `ImageAestheticFilterStage`
* **NSFW filtering**: Remove inappropriate content using `ImageNSFWFilterStage`
* **Automatic processing**: Filtering happens during the pipeline execution

### Pipeline with filtering

```python
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage

# Filter by aesthetic quality (keep images with score >= 0.5)
pipeline.add_stage(ImageAestheticFilterStage(
    model_dir="/models",
    score_threshold=0.5,  # Minimum aesthetic score
    num_gpus_per_worker=0.25,
))

# Filter NSFW content (keep images with score < 0.5)
pipeline.add_stage(ImageNSFWFilterStage(
    model_dir="/models", 
    score_threshold=0.5,  # Maximum NSFW score (images below this are kept)
    num_gpus_per_worker=0.25,
))
```

For custom filtering logic, you can create your own stage by extending `ProcessingStage[ImageBatch, ImageBatch]`.

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Filters <filters/index.md>
Embeddings <embeddings/index.md>
```
