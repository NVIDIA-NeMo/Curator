---
description: "Understand output directories, parquet embeddings, and packaging curated video data for training"
categories: ["video-curation"]
tags: ["export", "parquet", "webdataset", "metadata"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "howto"
modality: "video-only"
---

# Save and  Export

NeMo Curator writes clips, metadata, and embeddings to an output directory. Use these paths to integrate with training workflows.

## Output Layout

Common directories include:

- `clips/`: Encoded clip media (`.mp4`).
- `filtered_clips/`: Media for filtered-out clips.
- `metas/v0/`: Clip metadata (`.json`).
- `iv2_embd/`, `ce1_embd/`: Per-clip embeddings (`.pickle`).
- `iv2_embd_parquet/`, `ce1_embd_parquet/`: Parquet embeddings with columns `id` and `embedding`.
- `previews/`: Preview images (`.webp`).
- `processed_videos/`, `processed_clip_chunks/`: Video-level metadata and per-chunk statistics.

## Parquet Embeddings

The pipeline writes embeddings in Parquet batches to support scalable downstream processing and duplicate removal.

## Packaging

Package outputs as needed for training (for example, shard tar archives or a parquet index plus media files). Choose shard sizes that match your training I/O.
<!-- end -->
