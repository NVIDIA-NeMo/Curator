---
name: vid-caption
description: Generate text captions for video clips using Qwen VL vision-language model. Requires full GPU.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.1"
  modality: video
  gpu-required: "true"
  parent-skill: video
---

# Video Captioning Skill

Generate text descriptions for video clips using Qwen VL vision-language model. GPU required.

## When This Skill Applies

- User wants to generate captions for videos
- User mentions: "caption", "describe", "text description", "VLM"
- User is building video-text datasets

## GPU Requirements

- **CaptionPreparationStage**: CPU only (prepares inputs)
- **CaptionGenerationStage**: **1 full GPU** (Qwen VL model)
- **CaptionEnhancementStage**: **1 full GPU** (Qwen LM model, optional)
- A100/H100 recommended for large-scale captioning

## Important: Three-Stage Captioning Process

Video captioning requires multiple stages:

1. **CaptionPreparationStage** - Splits clips into windows and prepares inputs for the VL model
2. **CaptionGenerationStage** - Generates captions using Qwen VL
3. **CaptionEnhancementStage** (optional) - Enhances captions with more detail

## Skill Workflow

### Step 1: Understand the Goal

Ask the user:

1. What detail level? (brief vs detailed descriptions)
2. What domain? (general, autonomous vehicles, surveillance)
3. Are clips already extracted and transcoded?

### Step 2: Generate Pipeline Code

```python
# Video Captioning Pipeline
# GPU Required: Yes (1 full GPU for captioning)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU required for captioning")

print(f"GPU: {torch.cuda.get_device_name(0)}")

from nemo_curator.stages.video.io.video_reader import VideoReader
from nemo_curator.stages.video.caption.caption_preparation import CaptionPreparationStage
from nemo_curator.stages.video.caption.caption_generation import CaptionGenerationStage
from nemo_curator.stages.video.caption.caption_enhancement import CaptionEnhancementStage
from nemo_curator.pipeline import Pipeline

def generate_captions(
    input_dir: str,
    output_path: str,
    prompt_variant: str = "default",
    enhance: bool = False,
):
    """Generate video captions using Qwen VL.
    
    Args:
        input_dir: Directory containing video clips (must be transcoded MP4s)
        output_path: Path to save captioned results
        prompt_variant: "default", "av" (autonomous vehicle), or "av-surveillance"
        enhance: Whether to run caption enhancement stage
    """
    print(f"Input: {input_dir}")
    print(f"Prompt variant: {prompt_variant}")
    print(f"Enhancement: {enhance}")
    
    # Stage 1: Read video files
    reader = VideoReader(input_video_path=input_dir)
    
    # Stage 2: Prepare caption inputs (CPU stage)
    preparer = CaptionPreparationStage(
        model_variant="qwen",
        prompt_variant=prompt_variant,  # "default", "av", or "av-surveillance"
        sampling_fps=2.0,               # Frames per second to sample
        window_size=256,                # Frames per window
        verbose=False,
    )
    
    # Stage 3: Generate captions (GPU stage)
    captioner = CaptionGenerationStage(
        model_dir="models/qwen",
        model_variant="qwen",
        caption_batch_size=16,          # Batch size for inference
        fp8=False,                      # Use FP8 quantization for memory savings
        max_output_tokens=512,          # Maximum caption length
        verbose=False,
    )
    
    stages = [reader, preparer, captioner]
    
    # Stage 4: Enhance captions (optional, GPU stage)
    if enhance:
        enhancer = CaptionEnhancementStage(
            model_dir="models/qwen",
            model_variant="qwen",
            prompt_variant=prompt_variant,
            model_batch_size=128,
            max_output_tokens=512,
            verbose=False,
        )
        stages.append(enhancer)
    
    pipeline = Pipeline(name="video_captioning", stages=stages)
    results = pipeline.run()
    
    print(f"Captioned {len(results)} videos")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video/clip directory")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--prompt", default="default", 
                        choices=["default", "av", "av-surveillance"])
    parser.add_argument("--enhance", action="store_true", help="Enable caption enhancement")
    args = parser.parse_args()
    generate_captions(args.input, args.output, args.prompt, args.enhance)
```

## Available Stages

| Stage | Purpose | GPU | Parameters |
|-------|---------|-----|------------|
| `CaptionPreparationStage` | Prepare inputs for VL model | CPU | `prompt_variant`, `sampling_fps`, `window_size` |
| `CaptionGenerationStage` | Generate captions with Qwen VL | 1 GPU | `caption_batch_size`, `fp8`, `max_output_tokens` |
| `CaptionEnhancementStage` | Enhance existing captions | 1 GPU | `model_batch_size`, `max_output_tokens` |

## Prompt Variants

| Variant | Use Case | Description |
|---------|----------|-------------|
| `default` | General video | "Elaborate on the visual and narrative elements of the video in detail" |
| `av` | Autonomous vehicles | Focus on driving safety, cars, pedestrians, lane markers, road signs |
| `av-surveillance` | Traffic cameras | Focus on vehicle motion, driving safety from surveillance POV |

You can also provide a custom prompt via `prompt_text` parameter.

## Caption Output

Captions are stored in the clip's window objects:

```python
for task in results:
    for clip in task.data.clips:
        for window in clip.windows:
            # Basic caption from Qwen VL
            caption = window.caption.get("qwen")
            
            # Enhanced caption (if enhancement stage was used)
            enhanced = window.enhanced_caption.get("qwen_lm")
            
            print(f"Clip {clip.uuid}: {caption}")
```

## Execution

```bash
docker run --gpus all --rm \
    -v $(pwd)/clips:/input \
    -v $(pwd)/captions:/output \
    nvcr.io/nvidia/nemo-curator:latest \
    python /data/caption_pipeline.py --input /input --output /output --prompt default
```

## Prerequisites

Before running captioning, videos must be:

1. **Clipped** - Split into segments using `TransNetV2ClipExtractionStage` or `FixedStrideExtractorStage`
2. **Transcoded** - Encoded to standard format using `ClipTranscodingStage`

See the `vid-clip` skill for clipping workflows.

## Performance Tips

- Use `fp8=True` to reduce GPU memory usage (may slightly reduce quality)
- Adjust `caption_batch_size` based on GPU memory (lower = less memory)
- For large datasets, consider running captioning separately from other GPU stages
