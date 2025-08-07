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

1. Generate image embeddings for your dataset (for example, using `TimmImageEmbedder`)
2. Select and configure an image classifier (for example, Aesthetic or NSFW)
3. Apply the classifier to score or filter images based on the embeddings
4. Save or further process the classified dataset

You can use built-in classifiers or implement your own for advanced use cases.

---

## Available Classifiers

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

::: {grid-item-card} Aesthetic Classifier
:link: image-process-data-classifiers-aesthetic
:link-type: ref

Assess the subjective quality of images using a model trained on human aesthetic preferences. Useful for filtering or ranking images by visual appeal.
+++
{bdg-secondary}`Linear (MLP)` {bdg-secondary}`aesthetic_score`
:::

::: {grid-item-card} NSFW Classifier
:link: image-process-data-classifiers-nsfw
:link-type: ref

Detect not-safe-for-work (NSFW) content in images using a CLIP-based classifier. Helps remove or flag explicit material from your datasets.
+++
{bdg-secondary}`MLP (CLIP-based)` {bdg-secondary}`nsfw_score`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

aesthetic.md
nsfw.md
```
