---
description: "Provide an explicit JSON list of videos to load from a remote root"
categories: ["video-curation"]
tags: ["video", "load", "json", "s3", "client-partitioning", "fsspec"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "howto"
modality: "video-only"
---

(video-load-data-json-list)=

# Explicit File List (JSON)

For remote datasets, `ClientPartitioningStage` can use an explicit file list JSON. Each entry must be an absolute path under the specified root.

## JSON format

```json
[
  "s3://my-bucket/datasets/videos/video1.mp4",
  "s3://my-bucket/datasets/videos/video2.mkv",
  "s3://my-bucket/datasets/more_videos/video3.webm"
]
```

If any entry is outside the root, the stage raises an error.

## Python example

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.stages.client_partitioning import ClientPartitioningStage
from ray_curator.stages.video.io.video_reader import VideoReaderStage

ROOT = "s3://my-bucket/datasets/"
JSON_LIST = "s3://my-bucket/lists/videos.json"

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
