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

## Processing Options

Choose from the following stages to split, encode, filter, embed, caption, preview, and remove duplicates in your videos:

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Clip Videos
:link: video-process-clipping
:link-type: ref
Split long videos into shorter clips using fixed stride or scene-change detection.
+++
{bdg-primary}`clips`
{bdg-secondary}`fixed-stride`
{bdg-secondary}`transnetv2`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Encode Clips
:link: video-process-transcoding
:link-type: ref
Encode clips to H.264 using CPU or GPU encoders and tune performance.
+++
{bdg-primary}`clips`
{bdg-secondary}`h264_nvenc`
{bdg-secondary}`libopenh264`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Filter Clips and Frames
:link: video-process-filtering
:link-type: ref
Apply motion-based filtering and aesthetic filtering to improve dataset quality.
+++
{bdg-primary}`clips`
{bdg-primary}`frames`
{bdg-secondary}`motion`
{bdg-secondary}`aesthetic`
:::

:::{grid-item-card} {octicon}`device-camera;1.5em;sd-mr-1` Extract Frames
:link: video-process-frame-extraction
:link-type: ref
Extract frames from clips or full videos for embeddings, filtering, and analysis.
+++
{bdg-primary}`frames`
{bdg-secondary}`frames`
{bdg-secondary}`fps`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Create Embeddings
:link: video-process-embeddings
:link-type: ref
Generate clip-level embeddings with InternVideo2 or Cosmos-Embed1 for search and duplicate removal.
+++
{bdg-primary}`clips`
{bdg-secondary}`internvideo2`
{bdg-secondary}`cosmos-embed1`
:::

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Create Captions & Preview
:link: video-process-captions-preview
:link-type: ref
Produce clip captions and optional preview images for review workflows.
+++
{bdg-primary}`clips`
{bdg-primary}`frames`
{bdg-secondary}`captions`
{bdg-secondary}`preview`
:::

:::{grid-item-card} {octicon}`git-branch;1.5em;sd-mr-1` Remove Duplicate Embeddings
:link: video-process-dedup
:link-type: ref
Remove near-duplicates using semantic clustering and similarity with generated embeddings.
+++
{bdg-primary}`clips`
{bdg-secondary}`semantic`
{bdg-secondary}`pairwise`
:::

::::

## Write Outputs

Persist clips, embeddings, previews, and metadata at the end of the pipeline using `ClipWriterStage`. Refer to [Save & Export](video-save-export) for directory layout and examples.

Example (place as the final stage):

```python
from nemo_curator.stages.video.io.clip_writer import ClipWriterStage

pipeline.add_stage(
    ClipWriterStage(
        output_path=OUT_DIR,
        input_path=VIDEO_DIR,
        upload_clips=True,
        dry_run=False,
        generate_embeddings=True,
        generate_previews=False,
        generate_captions=False,
        embedding_algorithm="internvideo2",
        caption_models=[],
        enhanced_caption_models=[],
        verbose=True,
    )
)
```

Path helpers are available to resolve common locations (such as `clips/`, `filtered_clips/`, `previews/`, `metas/v0/`, and `iv2_embd_parquet/`).

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Clip Videos <clipping>
Encode Clips <transcoding>
Filter Clips and Frames <filtering>
Extract Frames <frame-extraction>
Create Embeddings <embeddings>
Create Captions & Preview <captions-preview>
Remove Duplicate Embeddings <dedup>
```
