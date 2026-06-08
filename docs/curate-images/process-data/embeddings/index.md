---
description: "Generate image embeddings using built-in embedders for classification, filtering, and similarity search"
categories: ["workflows"]
tags: ["embedding", "clip", "gpu-accelerated", "similarity-search"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "image-only"
---

(image-process-data-embeddings)=
# Image Embedding

Generate image embeddings for large-scale datasets using NeMo Curator's built-in embedders. Image embeddings enable downstream tasks such as classification, filtering, duplicate removal, and similarity search.

## How It Works

Image embedding in NeMo Curator typically follows these steps:

1. Load your dataset using `FilePartitioningStage` and `ImageReaderStage`
2. Configure the `ImageEmbeddingStage` with CLIP model settings
3. Apply the embedding stage to generate CLIP embeddings for each image
4. Continue with downstream processing stages (filtering, classification, etc.)

The embedding stage integrates seamlessly into NeMo Curator's pipeline architecture.

---

## Available Embedding Tools

::::{grid} 1 1 1 2
:gutter: 2

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` ImageEmbeddingStage
:link: clip-embedder
:link-type: doc
Generate CLIP embeddings using OpenAI's ViT-L/14 model for high-quality image representations.
:::

::::

---

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

CLIP ImageEmbeddingStage <clip-embedder.md>
```
