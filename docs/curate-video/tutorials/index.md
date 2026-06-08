---
description: "Collection of tutorials for video curation workflows including beginner guides and advanced pipeline customization techniques"
categories: ["video-curation"]
tags: ["tutorial", "video-processing", "pipeline", "customization", "workflow", "beginner"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "video-only"

---

(video-tutorials)=
# Video Curation Tutorials

Use the tutorials in this section to learn video curation with NeMo Curator.

```{tip}
Tutorials are organized by complexity and typically build on one another.
```

---

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Beginner Tutorial
:link: video-tutorials-beginner
:link-type: ref
Run your first splitting pipeline with the Python example, including model prep and common flags.
+++
{bdg-secondary}`video-splitting`
{bdg-secondary}`embeddings`
{bdg-secondary}`captioning`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Split and Deduplicate Videos
:link: video-tutorials-split-dedup
:link-type: ref
Split videos and then remove near-duplicates using KMeans + Pairwise semantic dedup.
+++
{bdg-secondary}`splitting`
{bdg-secondary}`semantic-deduplication`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Pipeline Customization Series
:link: video-tutorials-pipeline-cust-series
:link-type: ref
Customize pipelines by composing `ProcessingStage` classes and tuning resources.
+++
{bdg-secondary}`stages`
{bdg-secondary}`resources`
{bdg-secondary}`custom-pipelines`
:::

::::

```{toctree}
:hidden:
:maxdepth: 4

Beginner Tutorial <beginner>
Split and Deduplicate Videos <split-dedup>
Pipeline Customization <pipeline-customization/index>
```
