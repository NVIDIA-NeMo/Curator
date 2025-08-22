---
description: "Generate clip captions with Qwen and optional preview images"
categories: ["video-curation"]
tags: ["captions", "qwen", "preview", "video"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "howto"
modality: "video-only"
---

(video-process-captions-preview)=
# Captions & Preview

Prepare inputs, generate captions, optionally enhance them, and produce preview images.

## Preparation and previews

```python
from ray_curator.stages.video.caption.caption_preparation import CaptionPreparationStage
from ray_curator.stages.video.preview.preview import PreviewStage

prep = CaptionPreparationStage(
    model_variant="qwen",
    prompt_variant="default",
    prompt_text=None,
    sampling_fps=2.0,
    window_size=256,
    remainder_threshold=128,
    preprocess_dtype="float16",
    model_does_preprocess=False,
    generate_previews=True,
    verbose=True,
)
preview = PreviewStage(
    target_fps=1,
    target_height=240,
    verbose=True,
)
```

## Caption generation and enhancement

```python
from ray_curator.stages.video.caption.caption_generation import CaptionGenerationStage
from ray_curator.stages.video.caption.caption_enhancement import CaptionEnhancementStage

gen = CaptionGenerationStage(
    model_dir="/models",
    model_variant="qwen",
    caption_batch_size=8,
    fp8=False,
    max_output_tokens=512,
    model_does_preprocess=False,
    generate_stage2_caption=False,
    stage2_prompt_text=None,
    disable_mmcache=True,
)
enh = CaptionEnhancementStage(
    model_dir="/models",
    model_variant="qwen",
    prompt_variant="default",
    prompt_text=None,
    model_batch_size=128,
    fp8=False,
    max_output_tokens=512,
    verbose=True,
)
```

<!-- end -->
