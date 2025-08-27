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
# Clipping

Split long videos into shorter clips for downstream processing.

## Fixed stride

```python
from ray_curator.stages.video.clipping.clip_extraction_stages import FixedStrideExtractorStage

stage = FixedStrideExtractorStage(
    clip_len_s=10.0,
    clip_stride_s=10.0,
    min_clip_length_s=2.0,
    limit_clips=0,
)
```

## TransNetV2 scene-change detection

```python
from ray_curator.stages.video.clipping.video_frame_extraction import VideoFrameExtractionStage
from ray_curator.stages.video.clipping.transnetv2_extraction import TransNetV2ClipExtractionStage

frame_extractor = VideoFrameExtractionStage(
    decoder_mode="pynvc",  # or "ffmpeg_gpu", "ffmpeg_cpu"
    verbose=True,
)
transnet = TransNetV2ClipExtractionStage(
    model_dir="/models",
    threshold=0.4,
    min_length_s=2.0,
    max_length_s=10.0,
    max_length_mode="stride",  # or "truncate"
    crop_s=0.5,
    gpu_memory_gb=10.0,
    limit_clips=0,
    verbose=True,
)
```

<!-- end -->
