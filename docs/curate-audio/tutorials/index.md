---
description: "Hands-on tutorials for audio curation including beginner guides and pipeline customization"
categories: ["tutorials"]
tags: ["audio-tutorials", "hands-on", "pipeline-customization", "asr-inference", "quality-filtering"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "audio-only"
---

# Audio Curation Tutorials

Practice with guided, hands-on examples to build, customize, and run audio curation pipelines using NeMo Curator.

## Tutorial Categories

Learn audio curation through progressive tutorials that build from basic concepts to advanced customization.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Beginner Tutorial
:link: beginner
:link-type: doc
Create and run your first audio curation pipeline using the FLEURS dataset
+++
{bdg-primary}`beginner`
{bdg-secondary}`fleurs-dataset`
{bdg-secondary}`asr-inference`
{bdg-secondary}`quality-filtering`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Pipeline Customization
:link: pipeline-customization/index
:link-type: doc
Customize ASR models, quality filters, and integration workflows for your specific needs
+++
{bdg-secondary}`custom-models`
{bdg-secondary}`custom-filters`
{bdg-secondary}`integration`
{bdg-secondary}`advanced`
:::

::::

## What You'll Learn

These tutorials will teach you how to:

- Set up audio curation environments with NeMo Curator
- Load and process multilingual speech datasets
- Configure ASR inference with NeMo Framework models
- Implement quality assessment using WER and custom metrics
- Filter audio data based on transcription accuracy and duration
- Integrate audio processing with text curation workflows
- Customize pipelines for domain-specific audio datasets

## Prerequisites

Before starting these tutorials, ensure you have:

- NeMo Curator installed with audio support (`pip install nemo-curator[audio]`)
- Basic familiarity with Python and data processing concepts
- Access to NVIDIA GPU (recommended for ASR inference)
- Understanding of speech processing fundamentals

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Beginner Tutorial <beginner.md>
Pipeline Customization <pipeline-customization/index.md>
```

