---
description: "Customize audio curation pipelines with custom ASR models, quality filters, and integration workflows"
categories: ["tutorials"]
tags: ["pipeline-customization", "custom-models", "custom-filters", "advanced", "integration"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "audio-only"
---

(audio-tutorials-pipeline-cust-series)=
# Audio Pipeline Customization Tutorials

Learn how to customize NeMo Curator's audio curation pipelines for your specific needs. These tutorials cover custom ASR models, quality filters, and integration workflows.

## Customization Areas

Master advanced audio curation techniques through specialized customization tutorials:

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Custom ASR Models
:link: custom-asr-models
:link-type: doc
Integrate custom or fine-tuned ASR models into your curation pipeline
+++
{bdg-secondary}`custom-models`
{bdg-secondary}`fine-tuning`
{bdg-secondary}`domain-adaptation`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Custom Quality Filters
:link: custom-filters
:link-type: doc
Implement domain-specific quality assessment and filtering logic
+++
{bdg-secondary}`custom-filters`
{bdg-secondary}`quality-metrics`
{bdg-secondary}`domain-specific`
:::

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Integration Workflows
:link: integration-workflows
:link-type: doc
Build complex workflows combining audio, text, and multimodal processing
+++
{bdg-secondary}`multimodal`
{bdg-secondary}`workflow-integration`
{bdg-secondary}`advanced-pipelines`
:::

::::

## What You'll Learn

These customization tutorials will teach you how to:

- Integrate custom or fine-tuned ASR models into NeMo Curator
- Create domain-specific quality assessment metrics
- Build conditional processing workflows based on audio characteristics
- Combine audio curation with text and multimodal processing
- Optimize pipelines for specific hardware configurations
- Handle specialized audio formats and datasets

## Prerequisites

Before starting these tutorials, you should have:

- Completed the [Audio Beginner Tutorial](../beginner.md)
- Experience with Python programming and object-oriented design
- Understanding of speech processing concepts
- Familiarity with NeMo Curator's core architecture
- Access to development environment with GPU support

## Tutorial Progression

The customization tutorials build upon each other:

1. **[Custom ASR Models](custom-asr-models.md)** - Start with model customization
2. **[Custom Quality Filters](custom-filters.md)** - Add specialized filtering logic
3. **[Integration Workflows](integration-workflows.md)** - Combine with other modalities

Each tutorial includes practical examples, code samples, and performance considerations for production deployments.

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Custom ASR Models <custom-asr-models.md>
Custom Quality Filters <custom-filters.md>
Integration Workflows <integration-workflows.md>
```

