---
description: "Extract frames from clips or full videos for embeddings, filtering, and analysis"
categories: ["video-curation"]
tags: ["frames", "extraction", "fps", "ffmpeg", "nvdec"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "howto"
modality: "video-only"
---

(video-process-frame-extraction)=

# Frame Extraction

Extract frames from clips or full videos at target rates and resolutions. Use frames for embeddings (such as InternVideo2 and Cosmos‑Embed1), aesthetic filtering, previews, and custom analysis.

## Use Cases

- Prepare inputs for embedding models that expect frame sequences.
- Run aesthetic filtering that operates on sampled frames.
- Generate lightweight previews or QA snapshots.
- Provide frames for scene-change detection before clipping (TransNetV2).

## Before You Start

[Embeddings](video-process-embeddings) and [aesthetic filtering](video-process-filtering-aesthetic) require frames. If you need saved media files, frame extraction is optional.

---

## Quickstart

Use the pipeline stages or the example script flags to extract frames for embeddings, filtering, and analysis.

::::{tab-set}

:::{tab-item} Pipeline Stage

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.video.clipping.clip_extraction_stages import FixedStrideExtractorStage
from nemo_curator.stages.video.clipping.clip_frame_extraction import ClipFrameExtractionStage
from nemo_curator.utils.decoder_utils import FrameExtractionPolicy, FramePurpose
from nemo_curator.stages.video.embedding.internvideo2 import (
    InternVideo2FrameCreationStage,
    InternVideo2EmbeddingStage,
)

pipe = Pipeline(name="clip_frames_embeddings")
pipe.add_stage(FixedStrideExtractorStage(clip_len_s=10.0, clip_stride_s=10.0))
pipe.add_stage(
    ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        extract_purposes=(FramePurpose.EMBEDDINGS,),
        target_res=(-1, -1),
        verbose=True,
    )
)
pipe.add_stage(InternVideo2FrameCreationStage(model_dir="/models", target_fps=2.0, verbose=True))
pipe.add_stage(InternVideo2EmbeddingStage(model_dir="/models", gpu_memory_gb=20.0, verbose=True))
pipe.run()
```

:::

:::{tab-item} Script Flags

```bash
# Clip frames implicitly when generating embeddings or aesthetics
python -m nemo_curator.examples.video.video_split_clip_example \
  ... \
  --generate-embeddings \
  --clip-extraction-target-res -1

# Full-video frames for TransNetV2 scene change
python -m nemo_curator.examples.video.video_split_clip_example \
  ... \
  --splitting-algorithm transnetv2 \
  --transnetv2-frame-decoder-mode pynvc
```

:::
::::

## Options in NeMo Curator

NeMo Curator provides two complementary stages:

- `ClipFrameExtractionStage`: Extracts frames from already‑split clips. Supports several target FPS values and computes an LCM rate to reduce decode work.
- `VideoFrameExtractionStage`: Extracts frames from full videos (for example, before scene‑change detection). Supports PyNvCodec (NVDEC) or `ffmpeg` CPU/GPU decode.

### Extract Frames

::::{tab-set}
::: {tab-item} From Clips

```python
from nemo_curator.stages.video.clipping.clip_frame_extraction import (
    ClipFrameExtractionStage,
)
from nemo_curator.utils.decoder_utils import FrameExtractionPolicy, FramePurpose

extract_frames = ClipFrameExtractionStage(
    extraction_policies=(FrameExtractionPolicy.sequence,),
    extract_purposes=(FramePurpose.EMBEDDINGS,),  # sets default FPS if target_fps not provided
    target_res=(-1, -1),  # keep original resolution
    # target_fps=[1, 2],  # optional: override with explicit FPS values
    verbose=True,
)
```

:::
::: {tab-item} From Full Videos (Scene Change)

```python
from nemo_curator.stages.video.clipping.video_frame_extraction import VideoFrameExtractionStage

frame_extractor = VideoFrameExtractionStage(
    decoder_mode="pynvc",  # or "ffmpeg_gpu", "ffmpeg_cpu"
    output_hw=(27, 48),    # (height, width) for frame extraction
    pyncv_batch_size=64,   # batch size for PyNvCodec
    verbose=True,
)
```

:::
::::

## Parameters

```{list-table} Common Parameters
:header-rows: 1

* - Parameter
  - Description
* - `extraction_policies`
  - Frame selection strategy. Use `sequence` for uniform sampling. `middle` selects a single middle frame.
* - `target_fps`
  - For clips: sampling rate in frames per second. If you provide several integer values, the stage uses LCM sampling.
* - `extract_purposes`
  - Shortcut that sets default FPS for specific purposes (such as embeddings). You can still pass `target_fps` to override.
* - `target_res`
  - Output frame resolution `(height, width)`. Use `(-1, -1)` to keep original.
* - `num_cpus`
  - Number of CPU cores for frame extraction. Default: `3`.
* - `decoder_mode`
  - For full‑video extraction: `pynvc` (NVDEC), `ffmpeg_gpu`, or `ffmpeg_cpu`.
* - `output_hw`
  - For full‑video extraction: `(height, width)` tuple for frame dimensions. Default: `(27, 48)`.
* - `pyncv_batch_size`
  - For full‑video extraction: batch size for PyNvCodec processing. Default: `64`.
```

### LCM Sampling for Several FPS Values

If you provide several integer `target_fps` values (such as `1` and `2`), the clip stage decodes once at the LCM rate and then samples every k‑th frame to produce each target rate. This reduces decode cost.

```python
ClipFrameExtractionStage(
    extraction_policies=(FrameExtractionPolicy.sequence,),
    target_fps=[1, 2],  # LCM = 2; decode once at 2 FPS, then subsample
)
```

## Hardware and Performance

- Prefer `pynvc` (NVDEC) or `ffmpeg_gpu` for high throughput when GPU hardware is available; otherwise use `ffmpeg_cpu`.
- Use batching where applicable and track worker resource use.
- Keep resolution modest if memory limits apply; set `target_res` when needed.

## Downstream Dependencies

- **Embeddings**: InternVideo2 and Cosmos‑Embed1 expect frames at specific rates. Refer to [Embeddings](video-process-embeddings).
- **Aesthetic Filtering**: Requires frames extracted earlier. Refer to [Filtering](video-process-filtering).
- **Clipping with TransNetV2**: Uses full‑video frame extraction before scene‑change detection. Refer to [Clipping](video-process-clipping).

## Troubleshooting

- "Frame extraction failed": Check decoder mode and availability; confirm `ffmpeg` and drivers for GPU modes.
- Not enough frames for embeddings: Increase `target_fps` or adjust clip length; certain embedding stages can re‑extract at a higher rate when needed.
