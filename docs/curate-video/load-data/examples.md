---
description: "Examples for loading videos and extracting metadata with Ray Curator"
categories: ["video-curation"]
tags: ["video", "examples", "ray", "xenna", "pipeline"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "howto"
modality: "video-only"
---

# Python Examples

(video-load-data-examples)=

## Basic folder read

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.stages.video.io.video_reader import VideoReader

VIDEO_DIR = "/path/to/videos"  # or s3://bucket/path

pipe = Pipeline(name="video_read", description="Read videos and extract metadata")
pipe.add_stage(VideoReader(input_video_path=VIDEO_DIR, video_limit=-1, verbose=True))
pipe.run(XennaExecutor())
```

## Explicit list on remote storage

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.stages.client_partitioning import ClientPartitioningStage
from ray_curator.stages.video.io.video_reader import VideoReaderStage

ROOT = "s3://my-bucket/datasets/"  # root prefix
JSON_LIST = "s3://my-bucket/lists/videos.json"  # absolute paths under ROOT

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

<!-- end -->
