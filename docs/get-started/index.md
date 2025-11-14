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

Welcome to NeMo Curator! This toolkit enables you to curate large-scale datasets for training generative AI models across text, image, and video modalities.

**Who are these quickstarts for?**
- Data scientists and ML engineers who want to quickly test NeMo Curator's capabilities
- Users who want to run their first curation pipeline with minimal setup
- Anyone exploring NeMo Curator before committing to a full production deployment

**What you'll find here:**
Each quickstart below gets you up and running with a specific modality in under 30 minutes. They include basic installation, sample data, and a working example.

:::{tip}
For production deployments, cluster configurations, or detailed system requirements, see the [Setup & Deployment documentation](admin-overview).
:::

---

## Modality Quickstarts

The following quickstarts enable you to test out NeMo Curator for a given modality.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`typography;1.5em;sd-mr-1` Text Curation Quickstart
:link: gs-text
:link-type: ref
Set up your environment and run your first text curation pipeline with NeMo Curator. Learn how to install the toolkit, prepare your data, and use the pipeline architecture with modular stages to curate large-scale text datasets efficiently.

:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Image Curation Quickstart
:link: gs-image
:link-type: ref
Set up your environment and install NeMo Curator's image modules. Learn about prerequisites, installation methods, and how to use the toolkit to curate large-scale image-text datasets for generative model training.

:::

:::{grid-item-card} {octicon}`video;1.5em;sd-mr-1` Video Curation Quickstart
:link: gs-video
:link-type: ref
Set up your environment and run your first video curation pipeline. Learn about prerequisites, installation options, and how to split, encode, embed, and export curated clips at scale.

:::

::::
