---
description: "Load videos from local directories or cloud/object storage using partitioning stages"
categories: ["video-curation"]
tags: ["video", "load", "local", "s3", "gcs", "http", "file-partitioning"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "howto"
modality: "video-only"
---

(video-load-data-local-cloud)=

# Local and Cloud

Use `VideoReader` to load videos from local paths or remote URLs.

## Local paths

- Examples: `/data/videos/`, `/mnt/datasets/av/`
- Uses `FilePartitioningStage` to recursively discover files.
- Filters by extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`.
- Set `video_limit` to cap discovery during testing (`-1` means unlimited).

## Remote paths

- Examples: `s3://bucket/path/`, `gcs://bucket/path/`, `https://host/path/`
- Uses `ClientPartitioningStage` backed by `fsspec` to list files.
- Optional `input_list_json_path` allows explicit file lists under a root prefix.
- Wraps entries as `FSPath` for efficient byte access during reading.

## Python example

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.stages.video.io.video_reader import VideoReader

pipe = Pipeline(name="video_read", description="Read videos and extract metadata")
pipe.add_stage(VideoReader(input_video_path="s3://my-bucket/videos/", video_limit=100, verbose=True))
pipe.run(XennaExecutor())
```

<!-- end -->
