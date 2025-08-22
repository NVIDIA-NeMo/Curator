---
description: "Process video data by splitting into clips, encoding, generating embeddings and captions, and removing duplicates"
categories: ["video-curation"]
tags: ["splitting", "encoding", "embeddings", "captioning", "deduplication"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "video-only"
---

(video-process-data)=

# Process Data

Use NeMo Curator stages to split videos into clips, encode them, generate embeddings or captions, and remove duplicates.

## How it Works

Create a `Pipeline` and add stages for clip extraction, optional re-encoding and filtering, embeddings or captions, previews, and writing outputs. Each stage is modular and configurable to match your quality and performance needs.

## Options

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Clipping
:link: video-process-clipping
:link-type: ref
Split long videos into shorter clips using fixed stride or scene-change detection.
+++
{bdg-secondary}`fixed-stride`
{bdg-secondary}`transnetv2`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Filtering
:link: video-process-filtering
:link-type: ref
Apply motion-based filtering and aesthetic filtering to improve dataset quality.
+++
{bdg-secondary}`motion`
{bdg-secondary}`aesthetic`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Embeddings
:link: video-process-embeddings
:link-type: ref
Generate clip-level embeddings with InternVideo2 or Cosmos-Embed1 for search and duplicate removal.
+++
{bdg-secondary}`internvideo2`
{bdg-secondary}`cosmos-embed1`
:::

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Captions & Preview
:link: video-process-captions-preview
:link-type: ref
Produce clip captions and optional preview images for review workflows.
+++
{bdg-secondary}`captions`
{bdg-secondary}`preview`
:::

:::{grid-item-card} {octicon}`git-branch;1.5em;sd-mr-1` Duplicate Removal
:link: video-process-dedup
:link-type: ref
Remove near-duplicates using semantic clustering and similarity with generated embeddings.
+++
{bdg-secondary}`semantic`
{bdg-secondary}`pairwise`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Clipping <clipping>
Filtering <filtering>
Embeddings <embeddings>
Captions & Preview <captions-preview>
Duplicate Removal <dedup>
```
