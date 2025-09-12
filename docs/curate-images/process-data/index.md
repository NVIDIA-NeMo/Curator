---
description: "Process image data using embeddings, classifiers, and filtering for high-quality dataset curation"
categories: ["workflows"]
tags: ["data-processing", "embedding", "classification", "filtering", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "image-only"
---

(image-process-data)=
# Process Data for Image Curation

Process image data you've loaded into a WebDataset using NeMo Curator's suite of tools. These tools help you generate embeddings, classify images, and filter your dataset to prepare high-quality data for downstream AI tasks such as generative model training, dataset analysis, or quality control.

## How it Works

Image processing in NeMo Curator follows a pipeline-based approach with these stages:

1. **Partition files** using `FilePartitioningStage` to distribute WebDataset tar files
2. **Read images** using `ImageReaderStage` with DALI acceleration
3. **Generate embeddings** using `ImageEmbeddingStage` with CLIP models
4. **Apply classifiers** using `ImageAestheticFilterStage` and `ImageNSFWFilterStage`
5. **Save results** using `ImageWriterStage` to export curated datasets

Each stage processes `ImageBatch` objects containing images, metadata, and processing results. You can use built-in stages or implement custom stages for advanced use cases.

---

## Classifier Options

:::: {grid} 1 2 2 2
:gutter: 1 1 1 2

::: {grid-item-card} Aesthetic Filter Stage
:link: image-process-data-classifiers-aesthetic
:link-type: ref

Assess the subjective quality of images using a model trained on human aesthetic preferences. Filters images based on aesthetic score thresholds.
+++
{bdg-secondary}`ImageAestheticFilterStage` {bdg-secondary}`aesthetic_score`
:::

::: {grid-item-card} NSFW Filter Stage
:link: image-process-data-classifiers-nsfw
:link-type: ref

Detect not-safe-for-work (NSFW) content in images using a CLIP-based classifier. Filters explicit material from your datasets.
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

::: {grid-item-card} Custom Embedding Stage
:link: image-process-data-embeddings-custom
:link-type: ref

Implement your own image embedding logic by subclassing `ProcessingStage`. Useful for research models or custom pipelines.
+++
{bdg-secondary}`ProcessingStage` {bdg-secondary}`custom` {bdg-secondary}`advanced`
:::

::::

## Filtering Images

Filtering is built into the classification stages (`ImageAestheticFilterStage`, `ImageNSFWFilterStage`). Images that don't meet the specified thresholds are automatically filtered out during processing.

**Built-in filtering capabilities:**
* **Aesthetic filtering**: Remove images with low aesthetic scores using `ImageAestheticFilterStage`
* **NSFW filtering**: Remove inappropriate content using `ImageNSFWFilterStage`
* **Automatic processing**: Filtering happens during the pipeline execution

**Example: Pipeline with filtering**

```python
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage

# Filter by aesthetic quality (keep images with score >= 0.5)
pipeline.add_stage(ImageAestheticFilterStage(
    model_dir="/models",
    score_threshold=0.5,  # Minimum aesthetic score
    num_gpus_per_worker=0.25,
))

# Filter NSFW content (remove images with score >= 0.5)
pipeline.add_stage(ImageNSFWFilterStage(
    model_dir="/models", 
    score_threshold=0.5,  # Maximum NSFW score
    num_gpus_per_worker=0.25,
))
```

For custom filtering logic, you can implement your own `ProcessingStage` subclass.

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Classifiers <classifiers/index.md>
Embeddings <embeddings/index.md>
```
