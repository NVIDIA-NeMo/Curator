---
description: "Image classification tools including aesthetic and NSFW classifiers for dataset quality control"
categories: ["workflows"]
tags: ["classification", "aesthetic", "nsfw", "quality-filtering", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "image-only"
---

(image-process-data-classifiers)=
# Image Classifiers

NeMo Curator provides classifiers for image curation, including aesthetic and NSFW classifiers. These models help you filter, score, and curate large image datasets for downstream tasks such as generative model training and dataset quality control.

## How It Works

Image classification in NeMo Curator typically follows these steps:

1. Load images using `FilePartitioningStage` and `ImageReaderStage`
2. Generate image embeddings using `ImageEmbeddingStage`
3. Apply classification stages (for example, `ImageAestheticFilterStage` or `ImageNSFWFilterStage`)
4. Continue with further processing stages or save results

Classification stages integrate seamlessly into NeMo Curator's pipeline architecture.

---

## Available Classifiers

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

::: {grid-item-card} Aesthetic Filter Stage
:link: image-process-data-classifiers-aesthetic
:link-type: ref

Assess the subjective quality of images using a model trained on human aesthetic preferences. Filters images below a configurable aesthetic score threshold.
+++
{bdg-secondary}`ImageAestheticFilterStage` {bdg-secondary}`aesthetic_score`
:::

::: {grid-item-card} NSFW Filter Stage
:link: image-process-data-classifiers-nsfw
:link-type: ref

Detect not-safe-for-work (NSFW) content in images using a CLIP-based classifier. Filters images above a configurable NSFW probability threshold.
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
