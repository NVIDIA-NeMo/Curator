---
description: "Filter clips by motion or aesthetics to improve dataset quality"
categories: ["video-curation"]
tags: ["filtering", "motion", "aesthetic", "video"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "howto"
modality: "video-only"
---

(video-process-filtering)=

# Filtering

Apply motion-based filtering and aesthetic filtering to prune low-quality clips.

## Motion filtering

```python
from ray_curator.stages.video.filtering.motion_filter import MotionFilterStage
from ray_curator.stages.video.filtering.motion_vector_backend import MotionVectorDecodeStage

decode = MotionVectorDecodeStage(
    target_fps=2.0,
    target_duration_ratio=0.5,
    num_cpus_per_worker=4.0,
)
motion = MotionFilterStage(
    score_only=False,
    global_mean_threshold=0.00098,
    per_patch_min_256_threshold=0.000001,
    motion_filter_batch_size=64,
    num_gpus_per_worker=0.5,
    verbose=True,
)
```

## Aesthetic filtering

```python
from ray_curator.stages.video.filtering.clip_aesthetic_filter import ClipAestheticFilterStage

aesthetic = ClipAestheticFilterStage(
    model_dir="/models",
    score_threshold=3.5,
    reduction="min",  # or "mean"
    num_gpus_per_worker=0.25,
    verbose=True,
)
```

<!-- end -->
