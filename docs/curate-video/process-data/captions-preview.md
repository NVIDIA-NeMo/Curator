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

# Captions and Preview

Prepare inputs, generate captions, optionally enhance them, and produce preview images.

---

## Quickstart

Use the pipeline stages or the example script flags to prepare captions and preview images.

::::{tab-set}

:::{tab-item} Pipeline Stage

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.video.caption.caption_preparation import CaptionPreparationStage
from nemo_curator.stages.video.caption.caption_generation import CaptionGenerationStage
from nemo_curator.stages.video.caption.caption_enhancement import CaptionEnhancementStage
from nemo_curator.stages.video.preview.preview import PreviewStage

pipe = Pipeline(name="captions_preview")
pipe.add_stage(
    CaptionPreparationStage(
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
)
pipe.add_stage(PreviewStage(target_fps=1, target_height=240, verbose=True))
pipe.add_stage(
    CaptionGenerationStage(
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
)
pipe.run()
```

:::

:::{tab-item} Script Flags

```bash
python -m nemo_curator.examples.video.video_split_clip_example \
  ... \
  --generate-captions \
  --captioning-algorithm qwen \
  --captioning-window-size 256 \
  --captioning-remainder-threshold 128 \
  --captioning-sampling-fps 2.0 \
  --captioning-preprocess-dtype float16 \
  --captioning-batch-size 8 \
  --captioning-max-output-tokens 512 \
  --generate-previews \
  --preview-target-fps 1 \
  --preview-target-height 240
```

:::
::::

## Preparation and previews

1. Prepare caption inputs from each clip window. This step splits clips into fixed windows, formats model‑ready inputs for Qwen‑VL, and optionally stores per‑window `mp4` bytes for previews.

   ```python
   from nemo_curator.stages.video.caption.caption_preparation import CaptionPreparationStage
   from nemo_curator.stages.video.preview.preview import PreviewStage

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
   ```

2. Optionally generate `.webp` previews from each window’s `mp4` bytes for quick QA and review.

   ```python
   preview = PreviewStage(
       target_fps=1,
       target_height=240,
       verbose=True,
   )
   ```

### Parameters

::::{tab-set}

:::{tab-item} CaptionPreparationStage

```{list-table} Caption preparation parameters
:header-rows: 1
:widths: 22 12 12 54

* - Parameter
  - Type
  - Default
  - Description
* - `model_variant`
  - str
  - `"qwen"`
  - Vision‑language model used to format inputs for captioning (currently `qwen`).
* - `prompt_variant`
  - {"default", "av", "av-surveillance"}
  - `"default"`
  - Built‑in prompt to steer caption content when `prompt_text` is not provided.
* - `prompt_text`
  - str | None
  - `None`
  - Custom prompt text. When set, overrides `prompt_variant`.
* - `sampling_fps`
  - float
  - 2.0
  - Source sampling rate for creating per‑window inputs.
* - `window_size`
  - int
  - 256
  - Number of frames per window before captioning.
* - `remainder_threshold`
  - int
  - 128
  - Minimum leftover frames required to create a final shorter window.
* - `model_does_preprocess`
  - bool
  - `False`
  - Whether the downstream model performs its own preprocessing.
* - `preprocess_dtype`
  - str
  - `"float32"`
  - Data type for any preprocessing performed here.
* - `generate_previews`
  - bool
  - `True`
  - When `True`, return per‑window `mp4` bytes to enable preview generation.
* - `verbose`
  - bool
  - `False`
  - Log additional setup and per‑clip details.
```

:::
:::{tab-item} PreviewStage

```{list-table} Preview generation parameters
:header-rows: 1
:widths: 22 12 12 54

* - Parameter
  - Type
  - Default
  - Description
* - `target_fps`
  - float
  - 1.0
  - Frames per second for preview encoding.
* - `target_height`
  - int
  - 240
  - Output height in pixels; width auto‑scales to preserve aspect ratio.
* - `compression_level`
  - int (0–6)
  - 6
  - WebP compression level (`0` = lossless, higher = smaller files).
* - `quality`
  - int (0–100)
  - 50
  - WebP quality factor (`100` = best quality, larger files).
* - `num_cpus_per_worker`
  - float
  - 4.0
  - CPU threads mapped to `ffmpeg -threads` for encoding.
* - `verbose`
  - bool
  - `False`
  - Log warnings and per‑window encoding details.
```

:::

::::

## Caption generation and enhancement

1. Generate window‑level captions with a vision‑language model (Qwen‑VL). This stage reads `clip.windows[*].qwen_llm_input` created earlier and writes `window.caption["qwen"]`.

   ```python
   from nemo_curator.stages.video.caption.caption_generation import CaptionGenerationStage
   from nemo_curator.stages.video.caption.caption_enhancement import CaptionEnhancementStage

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

   ```

2. Optionally enhance captions with a text‑based LLM (Qwen‑LM) to expand and refine descriptions. This stage reads `window.caption["qwen"]` and writes `window.enhanced_caption["qwen_lm"]`.

   ```python
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

### Parameters

::::{tab-set}

:::{tab-item} CaptionGenerationStage

```{list-table} Caption generation parameters
:header-rows: 1
:widths: 22 12 12 54

* - Parameter
  - Type
  - Default
  - Description
* - `model_dir`
  - str
  - `"models/qwen"`
  - Directory for model weights; downloaded on each node if missing.
* - `model_variant`
  - {"qwen"}
  - `"qwen"`
  - Vision‑language model variant.
* - `caption_batch_size`
  - int
  - 16
  - Batch size for caption generation.
* - `fp8`
  - bool
  - `False`
  - Use FP8 weights when available.
* - `max_output_tokens`
  - int
  - 512
  - Maximum number of tokens to generate per caption.
* - `model_does_preprocess`
  - bool
  - `False`
  - Whether the model performs its own preprocessing.
* - `disable_mmcache`
  - bool
  - `False`
  - Disable multimodal cache for generation backends that support it.
* - `generate_stage2_caption`
  - bool
  - `False`
  - Enable a second‑pass caption for refinement.
* - `stage2_prompt_text`
  - str | None
  - `None`
  - Custom prompt for stage‑2 caption refinement.
* - `verbose`
  - bool
  - `False`
  - Emit additional logs during generation.
```

:::

:::{tab-item} CaptionEnhancementStage

```{list-table} Caption enhancement parameters
:header-rows: 1
:widths: 22 12 12 54

* - Parameter
  - Type
  - Default
  - Description
* - `model_dir`
  - str
  - `"models/qwen"`
  - Directory for language‑model weights; downloaded per node if missing.
* - `model_variant`
  - {"qwen"}
  - `"qwen"`
  - Language‑model variant.
* - `prompt_variant`
  - {"default", "av-surveillance"}
  - `"default"`
  - Built‑in enhancement prompt when `prompt_text` is not provided.
* - `prompt_text`
  - str | None
  - `None`
  - Custom enhancement prompt. When set, overrides `prompt_variant`.
* - `model_batch_size`
  - int
  - 128
  - Batch size for enhancement generation.
* - `fp8`
  - bool
  - `False`
  - Use FP8 weights when available.
* - `max_output_tokens`
  - int
  - 512
  - Maximum number of tokens to generate per enhanced caption.
* - `verbose`
  - bool
  - `False`
  - Emit additional logs during enhancement.
```

:::
::::

## Preview Generation

Generate lightweight `.webp` previews for each caption window to support review and QA workflows. A dedicated `PreviewStage` reads per-window `mp4` bytes and encodes WebP using `ffmpeg`.

### Preview Parameters

- `target_fps` (default `1.0`): Target frames per second for preview generation.
- `target_height` (default `240`): Output height. Width auto-scales to preserve aspect ratio.
- `compression_level` (range `0–6`, default `6`): WebP compression level. `0` is lossless; higher values reduce size with lower quality.
- `quality` (range `0–100`, default `50`): WebP quality. Higher values increase quality and size.
- `num_cpus_per_worker` (default `4.0`): Number of CPU threads mapped to `ffmpeg -threads`.
- `verbose` (default `False`): Emit more logs.

Behavior notes:

- If the input frame rate is lower than `target_fps` or the input height is lower than `target_height`, the stage logs a warning and preview quality can degrade.
- If `ffmpeg` fails, the stage logs the error and skips assigning preview bytes for that window.

### Example: Configure PreviewStage

```python
from nemo_curator.stages.video.preview.preview import PreviewStage

preview = PreviewStage(
    target_fps=1.0,
    target_height=240,
    compression_level=6,
    quality=50,
    num_cpus_per_worker=4.0,
    verbose=False,
)
```

### Outputs

The stage writes `.webp` files under the `previews/` directory that `ClipWriterStage` manages. Use the helper to resolve the path:

```python
from nemo_curator.stages.video.io.clip_writer import ClipWriterStage
previews_dir = ClipWriterStage.get_output_path_previews("/outputs")
```

Refer to Save & Export for directory structure and file locations: [Save & Export](video-save-export).

### Requirements and Troubleshooting

- `ffmpeg` with WebP (`libwebp`) support must be available in the environment.
- If you observe warnings about low frame rate or height, consider lowering `target_fps` or `target_height` to better match inputs.
- On encoding errors, check logs for the `ffmpeg` command and output to diagnose missing encoders.

<!-- end -->
