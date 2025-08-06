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

Image processing in NeMo Curator typically follows these steps:

1. **Load your dataset** using `ImageTextPairDataset`
2. **Generate image embeddings** using a built-in or custom embedder
3. **Apply classifiers** (such as aesthetic or NSFW) to score or filter images
4. **Filter images** based on classifier scores or metadata
5. **Save or export** your curated dataset for downstream use

You can use NeMo Curator's built-in tools or implement your own for advanced use cases.

---

## Classifier Options

:::: {grid} 1 2 2 2
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

## Embedding Options

:::: {grid} 1 2 2 2
:gutter: 1 1 1 2

::: {grid-item-card} TimmImageEmbedder
:link: image-process-data-embeddings-timm
:link-type: ref

Use state-of-the-art models from the PyTorch Image Models (timm) library for embedding generation. Highly recommended for most users.
+++
{bdg-secondary}`timm` {bdg-secondary}`vision transformer` {bdg-secondary}`CLIP`
:::

::: {grid-item-card} Custom ImageEmbedder
:link: image-process-data-embeddings-custom
:link-type: ref

Implement your own image embedding logic by subclassing the base class. Useful for research models or custom pipelines.
+++
{bdg-secondary}`custom` {bdg-secondary}`advanced`
:::

::::

## Filtering Images

Filter images in your dataset by applying thresholds to classifier scores (such as aesthetic or NSFW) or by using metadata fields. Unlike text curation, NeMo Curator does not currently provide built-in heuristic or content-based filters for images. Filtering is typically performed as a post-processing step after classification and embedding.

**Common filtering strategies:**
- Remove images with low aesthetic scores
- Remove or flag images with high NSFW scores
- Filter by metadata (e.g., resolution, aspect ratio)

**Example: Filtering by classifier score in Python**

```python
import dask_cudf
# Assume df is a Dask-cuDF DataFrame with 'aesthetic_score' and 'nsfw_score' columns
filtered = df[(df['aesthetic_score'] > 0.5) & (df['nsfw_score'] < 0.2)]
```

You can also implement custom filtering logic based on your project needs.

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Classifiers <classifiers/index.md>
Embeddings <embeddings/index.md>
```
