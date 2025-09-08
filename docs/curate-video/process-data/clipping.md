---
description: "Split videos into clips using fixed stride or TransNetV2 scene-change detection"
categories: ["video-curation"]
tags: ["clipping", "fixed-stride", "transnetv2", "video"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "howto"
modality: "video-only"
---

(video-process-clipping)=

# Video Clipping

Split long videos into shorter clips for downstream processing.

## How it Works

NeMo Curator provides two clipping stages: **Fixed Stride** and **TransNetV2** scene-change detection.

- Use Fixed Stride create uniform segments.
- Use TransNetV2 to cut at visual shot boundaries.

## Before You Start

Ensure inputs contain video bytes and basic metadata. The clipping stages require `video.source_bytes` to be present and metadata with `framerate` and `num_frames`.

---

## Quickstart

Use either the pipeline stages or the example script flags to create clips.

::::{tab-set}

:::{tab-item} Pipeline Stage

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.video.clipping.clip_extraction_stages import (
    FixedStrideExtractorStage,
)
from nemo_curator.stages.video.clipping.video_frame_extraction import (
    VideoFrameExtractionStage,
)
from nemo_curator.stages.video.clipping.transnetv2_extraction import (
    TransNetV2ClipExtractionStage,
)

pipe = Pipeline(name="clipping_examples")

# Fixed Stride
pipe.add_stage(
    FixedStrideExtractorStage(
        clip_len_s=10.0,
        clip_stride_s=10.0,
        min_clip_length_s=2.0,
        limit_clips=0,
    )
)

# TransNetV2 (requires full-video frame extraction first)
pipe.add_stage(VideoFrameExtractionStage(decoder_mode="pynvc", verbose=True))
pipe.add_stage(
    TransNetV2ClipExtractionStage(
        model_dir="/models",
        threshold=0.4,
        min_length_s=2.0,
        max_length_s=10.0,
        max_length_mode="stride",
        crop_s=0.5,
        gpu_memory_gb=10.0,
        limit_clips=-1,
        verbose=True,
    )
)

pipe.run()
```

:::

:::{tab-item} Script Flags

```bash
# Fixed stride
python -m nemo_curator.examples.video.video_split_clip_example \
  ... \
  --splitting-algorithm fixed_stride \
  --fixed-stride-split-duration 10.0 \
  --fixed-stride-min-clip-length-s 2.0 \
  --limit-clips 0

# TransNetV2
python -m nemo_curator.examples.video.video_split_clip_example \
  ... \
  --splitting-algorithm transnetv2 \
  --transnetv2-frame-decoder-mode pynvc \
  --transnetv2-threshold 0.4 \
  --transnetv2-min-length-s 2.0 \
  --transnetv2-max-length-s 10.0 \
  --transnetv2-max-length-mode stride \
  --transnetv2-crop-s 0.5 \
  --transnetv2-gpu-memory-gb 10.0 \
  --limit-clips 0
```

:::
::::

## Clipping Options

### Fixed Stride

The `FixedStrideExtractorStage` steps through the video duration by `clip_stride_s`, creating spans of length `clip_len_s` (it truncates the final span at the video end when needed). It filters spans shorter than `min_clip_length_s` and appends `Clip` objects identified by source and frame indices.

```python
from nemo_curator.stages.video.clipping.clip_extraction_stages import FixedStrideExtractorStage

stage = FixedStrideExtractorStage(
    clip_len_s=10.0,
    clip_stride_s=10.0,
    min_clip_length_s=2.0,
    limit_clips=0,
)
```

:::{tip} If `limit_clips > 0` and the `Video` already has clips, the stage skips processing. It does not cap the number of clips generated within the same run.
:::

### TransNetV2 Scene-Change Detection

TransNetV2 is a shot-boundary detection model that identifies transitions between shots. The stage converts those transitions into scenes, applies length/crop rules, and emits clips aligned to scene boundaries.

Using extracted frames of size 27×48×3, the model predicts shot transitions, converts them into scenes, and applies filtering: `min_length_s`, `max_length_s` with `max_length_mode` ("truncate" or "stride"), and optional `crop_s` at both ends. It creates `Clip` objects for the resulting spans, then stops after it reaches `limit_clips` (> 0), and releases frames from memory after processing.

1. Run `VideoFrameExtractionStage` first to populate `video.frame_array`.

   ```python
   from nemo_curator.stages.video.clipping.video_frame_extraction import VideoFrameExtractionStage
   from nemo_curator.stages.video.clipping.transnetv2_extraction import TransNetV2ClipExtractionStage

   frame_extractor = VideoFrameExtractionStage(
       decoder_mode="pynvc",  # or "ffmpeg_gpu", "ffmpeg_cpu"
       verbose=True,
   )
   ```

   :::{important}
   Frames must be `(27, 48, 3)` per frame; the stage accepts arrays shaped `(num_frames, 27, 48, 3)` and transposes from `(48, 27, 3)` automatically.
   :::

1. Configure TransNetV2 and run the stage in your pipeline to generate clips from the detected scenes.

   ```python
   transnet = TransNetV2ClipExtractionStage(
       model_dir="/models",
       threshold=0.4,
       min_length_s=2.0,
       max_length_s=10.0,
       max_length_mode="stride",  # or "truncate"
       crop_s=0.5,
       gpu_memory_gb=10.0,
       limit_clips=-1,
       verbose=True,
   )
   ```
