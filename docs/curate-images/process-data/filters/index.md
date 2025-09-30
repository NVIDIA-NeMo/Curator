---
description: "Image filtering tools including aesthetic and NSFW filters for dataset quality control"
categories: ["workflows"]
tags: ["filtering", "aesthetic", "nsfw", "quality-filtering", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "image-only"
---

(image-process-data-filters)=

# Image Filters

NeMo Curator provides filters for image curation, including aesthetic and NSFW filters. These models help you filter, score, and curate large image datasets for downstream tasks such as generative model training and dataset quality control.

## How It Works

Image filtering in NeMo Curator typically follows these steps:

1. Load images using `FilePartitioningStage` and `ImageReaderStage`
2. Generate image embeddings using `ImageEmbeddingStage`
3. Apply filtering stages (for example, `ImageAestheticFilterStage` or `ImageNSFWFilterStage`)
4. Continue with further processing stages or save results

Filtering stages integrate seamlessly into NeMo Curator's pipeline architecture.

## Prerequisites

Before using filtering stages, ensure that:

- Load images using `ImageReaderStage`
- Generate image embeddings using `ImageEmbeddingStage`
- Populate the `ImageObject.embedding` field for each image

## Imports

```python
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
```

---

## Available Filters

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

::: {grid-item-card} Aesthetic Filter Stage
:link: image-process-data-filters-aesthetic
:link-type: ref

Assess the subjective quality of images using a model trained on human aesthetic preferences. Filters images below a configurable aesthetic score threshold (0.0 to 1.0).
+++
{bdg-secondary}`ImageAestheticFilterStage` {bdg-secondary}`aesthetic_score`
:::

::: {grid-item-card} NSFW Filter Stage
:link: image-process-data-filters-nsfw
:link-type: ref

Detect not-safe-for-work (NSFW) content in images using a CLIP-based filter. Removes images above a configurable NSFW probability threshold (0.0 to 1.0).
+++
{bdg-secondary}`ImageNSFWFilterStage` {bdg-secondary}`nsfw_score`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

aesthetic.md
nsfw.md
```
