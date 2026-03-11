---
description: "Quickstart guides for getting started with NeMo Curator across text, image, and video modalities with minimal setup"
categories: ["getting-started"]
tags: ["quickstart", "installation", "python-api", "tutorial"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "universal"
---

(gs-overview)=
# About Getting Started

## Before You Start

Welcome to NeMo Curator! This framework streamlines the curation and pre-processing of large-scale datasets for training generative AI models across text, image, audio and video modalities.

**Who are these quickstarts for?**
- AI/ML engineers and researchers who want to quickly test NeMo Curator's capabilities
- Users looking to run an initial curation pipeline with minimal setup
- Individuals exploring NeMo Curator prior to a full production deployment

**What you'll find here:**
Each quickstart enables you to get started with a specific domain in less than 30 minutes. Quickstarts provide basic installation steps, sample data, and a working example.

:::{tip}
For production deployments, cluster configurations, or detailed system requirements, refer to the [Setup & Deployment documentation](admin-overview).
:::

---

## Modality Quickstarts

The following quickstarts allow you to test NeMo Curator using a selected data modality.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`typography;1.5em;sd-mr-1` Text Curation Quickstart
:link: gs-text
:link-type: ref
Set up your environment and execute your first text curation pipeline with NeMo Curator. Instructions cover installation, data preparation, and use of the modular pipeline architecture for efficient large-scale text dataset curation.

:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Image Curation Quickstart
:link: gs-image
:link-type: ref
Set up your environment and install the NeMo Curator image modules. The quickstart explains prerequisites, installation methods, and the use of the framework to curate large-scale image-text datasets for generative AI model training.

:::

:::{grid-item-card} {octicon}`video;1.5em;sd-mr-1` Video Curation Quickstart
:link: gs-video
:link-type: ref
Set up your environment and execute your first video curation pipeline. The instructions include prerequisites, installation options, and guidance on splitting, encoding, embedding, and exporting curated video clips at scale.

:::

:::{grid-item-card} {octicon}`audio;1.5em;sd-mr-1` Audio Curation Quickstart
:link: gs-audio
:link-type: ref
Set up your environment and execute your first audio curation pipeline with NeMo Curator. Instructions cover installation, data preparation, and use of the modular pipeline architecture for efficient large-scale audio speech dataset curation.

::::
