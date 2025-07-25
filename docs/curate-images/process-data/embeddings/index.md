---
description: "Generate image embeddings using built-in and custom embedders for classification, filtering, and similarity search"
categories: ["workflows"]
tags: ["embedding", "timm", "custom", "gpu-accelerated", "similarity-search"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "image-only"
---

(image-process-data-embeddings)=
# Image Embedding

Generate image embeddings for large-scale datasets using NeMo Curator's built-in and custom embedders. Image embeddings enable downstream tasks such as classification, filtering, duplicate removal, and similarity search.

## How It Works

Image embedding in NeMo Curator typically follows these steps:

1. Load your dataset using `ImageTextPairDataset`
2. Select and configure an embedder (for example, `TimmImageEmbedder`)
3. Apply the embedder to generate embeddings for each image
4. Save the resulting dataset with embeddings for downstream use

You can use built-in embedders or implement your own for advanced use cases.

---

## Available Embedding Tools

::::{grid} 1 1 1 2
:gutter: 2

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` TimmImageEmbedder
:link: timm
:link-type: doc
Use state-of-the-art models from the PyTorch Image Models (timm) library for embedding generation
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Custom ImageEmbedder
:link: custom
:link-type: doc
Implement your own image embedding logic by subclassing the base class
:::

::::

---

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

TimmImageEmbedder <timm.md>
Custom ImageEmbedder <custom.md>
```
