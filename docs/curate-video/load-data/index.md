---
description: "Load video data into NeMo Curator from local paths or fsspec-supported storage, including explicit file list support"
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
5. Set `video_limit` to cap discovery; use `None` for unlimited. Set `verbose=True` to log detailed per-video information.

---

(video-load-data-local-cloud)=

## Local and Cloud

Use `VideoReader` to load videos from local paths or remote URLs.

### Local Paths

- Examples: `/data/videos/`, `/mnt/datasets/av/`
- Uses `FilePartitioningStage` to recursively discover files.
- Filters by extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`.
- Set `video_limit` to cap discovery during testing (`None` means unlimited).

### Remote Paths

- Examples: `s3://bucket/path/`, `gcs://bucket/path/`, `https://host/path/`, and other fsspec-supported protocols such as `s3a://` and `abfs://`.
- Uses `ClientPartitioningStage` backed by `fsspec` to list files.
- Optional `input_list_json_path` allows explicit file lists under a root prefix.
- Wraps entries as `FSPath` for efficient byte access during reading.

```{tip}
Use an object storage prefix (for example, `s3://my-bucket/videos/`) to stream from cloud storage. Configure credentials in your environment or client configuration.
```

### Example

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.video.io.video_reader import VideoReader

pipe = Pipeline(name="video_read", description="Read videos and extract metadata")
pipe.add_stage(VideoReader(input_video_path="s3://my-bucket/videos/", video_limit=None, verbose=True))
pipe.run()
```

(video-load-data-json-list)=

## Explicit File List (JSON)

For remote datasets, `ClientPartitioningStage` can use an explicit file list JSON. Each entry must be an absolute path under the specified root.

### JSON Format

```json
[
  "s3://my-bucket/datasets/videos/video1.mp4",
  "s3://my-bucket/datasets/videos/video2.mkv",
  "s3://my-bucket/datasets/more_videos/video3.webm"
]
```

If any entry is outside the root, the stage raises an error.

### Example

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.client_partitioning import ClientPartitioningStage
from nemo_curator.stages.video.io.video_reader import VideoReaderStage

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
pipe.run()
```

## Supported File Types

The loader filters these video extensions by default:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.webm`

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
