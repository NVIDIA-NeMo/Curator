---
description: "Load video data into NeMo Curator from local paths or S3-compatible storage, including explicit file list support"
categories: ["video-curation"]
tags: ["video", "load", "s3", "local", "file-list"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "howto"
modality: "video-only"
---

(video-load-data)=

# Video Data Loading

Load video data for curation using NeMo Curator.

## How it Works

NeMo Curator loads videos with a composite stage that discovers files and extracts metadata:

1. `VideoReader` decomposes into a partitioning stage plus a reader stage.
2. Local paths use `FilePartitioningStage` to list files; remote URLs (for example, `s3://`, `gcs://`, `http(s)://`) use `ClientPartitioningStage` backed by `fsspec`.
3. For remote datasets, you can optionally supply an explicit file list using `ClientPartitioningStage.input_list_json_path`.
4. `VideoReaderStage` downloads bytes (local or via `FSPath`) and calls `video.populate_metadata()` to extract resolution, fps, duration, encoding format, and other fields.
5. Set `video_limit` to cap discovery during tests; set `verbose=True` to log detailed per-video information.

## Options

:::::{grid} 1 1 1 2
:gutter: 1 1 1 2

::::{grid-item-card} {octicon}`device-camera-video;1.5em;sd-mr-1` Local and Cloud
::link: video-load-data-local-cloud
::link-type: ref
Load from local directories or cloud/object storage using file partitioning.
+++
{bdg-secondary}`local`
{bdg-secondary}`s3`
{bdg-secondary}`gcs`
::::

::::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Explicit File List (JSON)
::link: video-load-data-json-list
::link-type: ref
Provide a JSON list of files under a root prefix for precise control.
+++
{bdg-secondary}`json`
{bdg-secondary}`client-partitioning`
::::

::::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Python Examples
::link: video-load-data-examples
::link-type: ref
End-to-end examples with `Pipeline`, `VideoReader`, and `XennaExecutor`.
+++
{bdg-secondary}`python`
{bdg-secondary}`ray-curator`
::::

:::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Local and Cloud <local-cloud>
Explicit File List (JSON) <json-list>
Examples <examples>
```

## Supported File Types

The loader filters these video extensions by default:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.webm`

## Local vs Remote Paths

- Local paths (for example, `/data/videos/`) use local file partitioning.
- Remote URLs (for example, `s3://bucket/path/`, `gcs://bucket/path/`, `http(s)://...`) use client-side partitioning. The loader detects remote paths using the `fsspec` protocol for the URL.

```{tip}
Use an S3 prefix (for example, `s3://my-bucket/videos/`) to stream from object storage. Configure credentials in your environment or client configuration.
```

## JSON File List Format

When using client-side partitioning, supply a JSON list of absolute paths under the provided root. For example, if your `file_paths` is `s3://my-bucket/datasets/`, then each entry must start with that prefix:

```json
[
  "s3://my-bucket/datasets/videos/video1.mp4",
  "s3://my-bucket/datasets/videos/video2.mkv",
  "s3://my-bucket/datasets/more_videos/video3.webm"
]
```

Use this with `ClientPartitioningStage.input_list_json_path` in a custom pipeline. The stage validates that every entry is under the root and converts entries to relative paths internally.

## Examples  

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.stages.video.io.video_reader import VideoReader

VIDEO_DIR = "/path/to/videos"  # or s3://bucket/path

pipe = Pipeline(name="video_read", description="Read videos and extract metadata")
pipe.add_stage(VideoReader(input_video_path=VIDEO_DIR, video_limit=-1, verbose=True))
pipe.run(XennaExecutor())
```

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.stages.client_partitioning import ClientPartitioningStage
from ray_curator.stages.video.io.video_reader import VideoReaderStage

ROOT = "s3://my-bucket/datasets/"  # root prefix
JSON_LIST = "s3://my-bucket/lists/videos.json"  # contains relative paths under ROOT

pipe = Pipeline(name="video_read_json_list", description="Read specific videos via JSON list")
pipe.add_stage(
    ClientPartitioningStage(
        file_paths=ROOT,
        input_list_json_path=JSON_LIST,
        files_per_partition=1,
        file_extensions=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
    )
)
pipe.add_stage(VideoReaderStage(verbose=True))
pipe.run(XennaExecutor())
```

## Metadata on Load

After a successful read, the loader populates the following metadata fields for each video:

- `size` (bytes)
- `width`, `height`
- `framerate`
- `num_frames`
- `duration` (seconds)
- `video_codec`, `pixel_format`, `audio_codec`
- `bit_rate_k`

```{note}
With `verbose=True`, the loader logs size, resolution, fps, duration, weight, and bit rate for each processed video.
```
<!-- end -->
