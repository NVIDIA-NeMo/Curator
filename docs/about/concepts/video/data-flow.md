---
description: "Understanding data flow in video curation pipelines including Ray object store and streaming optimization"
categories: ["concepts-architecture"]
tags: ["data-flow", "distributed", "ray", "streaming", "performance", "video-curation"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "video-only"
---

(about-concepts-video-data-flow)=

# Data Flow

Understanding how data moves through NeMo Curator's video curation pipelines is key to optimizing performance and resource usage.

- Data moves between stages via Ray's distributed object store, enabling efficient, in-memory transfer between distributed actors.
- In streaming mode, the executor returns final stage outputs while intermediate state stays in memory, reducing I/O overhead and improving throughput.
- The auto-scaling component continuously balances resources to maximize pipeline throughput, dynamically allocating workers to stages as needed.
- Writer stages persist outputs at the end of the pipeline, including clip media, embeddings (pickle and parquet variants), and metadata JSON files.

This architecture enables efficient processing of large-scale video datasets, with minimal data movement and optimal use of available hardware.

## Writer Output Layout

Writer stages produce the following directories under the configured output path:

- `clips/`: MP4 clip files
- `filtered_clips/`: MP4 files for filtered clips
- `previews/`: WebP preview images for windows
- `metas/v0/`: Per-clip JSON metadata
- `iv2_embd/`: Per-clip embeddings (pickle) for InternVideo2
- `iv2_embd_parquet/`: Per-video embeddings (parquet) for InternVideo2
- `ce1_embd/`: Per-clip embeddings (pickle) for Cosmos-Embed1
- `ce1_embd_parquet/`: Per-video embeddings (parquet) for Cosmos-Embed1
- `processed_videos/`: Per-video JSON metadata
- `processed_clip_chunks/`: Per-clip-chunk JSON statistics
