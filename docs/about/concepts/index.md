---
description: "Core concepts and terminology for NeMo Curator across text, image, and video data curation modalities"
categories: ["concepts-architecture"]
tags: ["concepts", "fundamentals", "multimodal", "architecture"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(about-concepts)=
# Concepts

Learn about the core components and concepts introduced by NeMo Curator. The following concepts are organized by each major modality.

## Modality Concepts

Learn about working with specific modalities using NeMo Curator.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Image Curation Concepts
:link: about-concepts-image
:link-type: ref

Explore key concepts for image data curation, including scalable loading, processing (embedding, classification, filtering, deduplication), and dataset export.
:::

:::{grid-item-card} {octicon}`typography;1.5em;sd-mr-1` Text Curation Concepts
:link: about-concepts-text
:link-type: ref

Learn about text data curation, covering data loading, processing (filtering, deduplication, classification), and synthetic data generation.
:::

:::{grid-item-card} {octicon}`video;1.5em;sd-mr-1` Video Curation Concepts
:link: about-concepts-video
:link-type: ref

Discover video data curation concepts, such as distributed processing, pipeline stages, execution modes, and efficient data flow.
:::

:::{grid-item-card} {octicon}`unmute;1.5em;sd-mr-1` Audio Curation Concepts
:link: about-concepts-audio
:link-type: ref

Learn about speech data curation, ASR inference, quality assessment, and audio-text integration workflows.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

Image Concepts <image/index.md>
Text Concepts <text/index.md>
Video Concepts <video/index.md>
Audio Concepts <audio/index.md>
```
