---
description: "Hands-on tutorials for image curation workflows using NeMo Curator tools and techniques"
categories: ["tutorials"]
tags: ["tutorials", "image-curation", "hands-on", "examples"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "image-only"
---

(image-tutorials)=
# Image Curation Tutorials


This section contains practical tutorials that show how to use NVIDIA NeMo Curator for various image curation tasks. Each tutorial provides step-by-step guidance for specific use cases.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Beginner Tutorial
:link: image-tutorials-beginner
:link-type: ref
Get started with basic image curation using NeMo Curator. Learn how to create pipelines that filter images by aesthetic quality and NSFW content.
+++
{bdg-primary}`beginner`
{bdg-secondary}`quality-filtering`
{bdg-secondary}`nsfw-detection`
:::

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Image Duplicate Removal Workflow
:link: image-tutorials-dedup
:link-type: ref
Learn how to remove duplicate and near-duplicate images using semantic duplicate removal with CLIP embeddings and clustering.
+++
{bdg-secondary}`duplicate-removal`
{bdg-secondary}`embeddings`
{bdg-secondary}`clustering`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Beginner Tutorial <beginner>
Image Duplicate Removal Workflow <dedup-workflow>
```

## Resources

For working Python examples and command-line scripts, refer to the [Tutorials directory](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/image) in the [NeMo Curator GitHub repository](https://github.com/NVIDIA-NeMo/Curator).
