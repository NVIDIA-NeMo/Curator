---
description: "Understand output directories, parquet embeddings, and packaging curated video data for training"
categories: ["video-curation"]
tags: ["export", "parquet", "webdataset", "metadata"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "howto"
modality: "video-only"
---

(video-save-export)=

# Save and Export

NeMo Curator writes clips, metadata, previews, and embeddings to a structured output directory. Use this guide to add the writer to your pipeline, understand the directories it creates, and prepare artifacts for training.

## Writer Stage

Use `ClipWriterStage` as the final stage in your pipeline.

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
        embedding_algorithm="cosmos-embed1",  # or "internvideo2"
        caption_models=["qwen"],
        enhanced_caption_models=["qwen_lm"],
        verbose=True,
    )
)
```

### Parameters

```{list-table}
:header-rows: 1

* - Parameter
  - Type
  - Description
* - `output_path`
  - `str`
  - Base directory or URI for outputs.
* - `input_path`
  - `str`
  - Root of input videos; used to derive processed metadata paths. Must be a prefix of input video paths.
* - `upload_clips`
  - `bool`
  - Write `.mp4` clips to `clips/` and filtered clips to `filtered_clips/`.
* - `dry_run`
  - `bool`
  - Skip writing clip bytes, preview images, embeddings, and per-clip metadata. The stage still writes video-level and chunk-level metadata.
* - `generate_embeddings`
  - `bool`
  - When true, the stage logs errors if embeddings for the selected algorithm are missing. When embeddings exist, the stage writes per-clip pickles and per-chunk Parquet files.
* - `generate_previews`
  - `bool`
  - When true, the stage logs errors for missing preview bytes and writes `.webp` images when present.
* - `generate_captions`
  - `bool`
  - The stage includes captions in metadata when upstream stages provide them.
* - `embedding_algorithm`
  - `str`
  - Accepted: `cosmos-embed1` or `internvideo2`. Default: `cosmos-embed1`.
* - `caption_models`
  - `list[str] | None`
  - Ordered caption models to emit. Use `[]` when not using captions.
* - `enhanced_caption_models`
  - `list[str] | None`
  - Ordered enhancement models to emit. Use `[]` when not using enhanced captions.
* - `verbose`
  - `bool`
  - Emit detailed logs.
* - `max_workers`
  - `int`
  - Thread pool size for writing.
* - `log_stats`
  - `bool`
  - Reserved for future detailed stats logging.
```

## Output Directories

The writer produces these directories under `output_path`:

- `clips/`: Encoded clip media (`.mp4`).
- `filtered_clips/`: Media for filtered-out clips.
- `previews/`: Preview images (`.webp`).
- `metas/v0/`: Per-clip metadata (`.json`).
- `iv2_embd/`, `ce1_embd/`: Per-clip embeddings (`.pickle`).
- `iv2_embd_parquet/`, `ce1_embd_parquet/`: Parquet batches with columns `id` and `embedding`.
- `processed_videos/`, `processed_clip_chunks/`: Video-level metadata and per-chunk statistics.

### Per-Clip Metadata

Each clip writes a JSON file under `metas/v0/` with clip- and window-level fields:

```json
{
  "span_uuid": "d2d0b3d1-...",
  "source_video": "/data/videos/vid.mp4",
  "duration_span": [0.0, 5.0],
  "width_source": 1920,
  "height_source": 1080,
  "framerate_source": 30.0,
  "clip_location": "/outputs/clips/d2/d2d0b3d1-....mp4",
  "motion_score": { "global_mean": 0.51, "per_patch_min_256": 0.29 },
  "aesthetic_score": 0.72,
  "windows": [
    {
      "start_frame": 0,
      "end_frame": 30,
      "qwen_caption": "A person walks across a room",
      "qwen_lm_enhanced_caption": "A person briskly crosses a bright modern room"
    }
  ],
  "valid": true
}
```

- Caption keys follow `<model>_caption` and `<model>_enhanced_caption`, based on `caption_models` and `enhanced_caption_models`.
- With `dry_run=True`, per-clip metadata is not written. Video- and chunk-level metadata are still written.
- The stage writes video-level metadata and per-chunk stats to `processed_videos/` and `processed_clip_chunks/`.

### Embeddings and Parquet outputs

- When embeddings exist, the stage writes per-clip `.pickle` files under `iv2_embd/` or `ce1_embd/`.
- The stage also batches embeddings per clip chunk into Parquet files under `iv2_embd_parquet/` or `ce1_embd_parquet/` with columns `id` and `embedding` and writes those files to disk.

## Helpers

### Resolve Paths Programmatically

Use helpers to construct paths consistently:

```python
from nemo_curator.stages.video.io.clip_writer import ClipWriterStage

OUT = "/outputs"

clips_dir = ClipWriterStage.get_output_path_clips(OUT)
filtered_clips_dir = ClipWriterStage.get_output_path_clips(OUT, filtered=True)
previews_dir = ClipWriterStage.get_output_path_previews(OUT)
metas_dir = ClipWriterStage.get_output_path_metas(OUT, "v0")
iv2_parquet_dir = ClipWriterStage.get_output_path_iv2_embd_parquet(OUT)
ce1_parquet_dir = ClipWriterStage.get_output_path_ce1_embd_parquet(OUT)
processed_videos_dir = ClipWriterStage.get_output_path_processed_videos(OUT)
processed_chunks_dir = ClipWriterStage.get_output_path_processed_clip_chunks(OUT)
```
