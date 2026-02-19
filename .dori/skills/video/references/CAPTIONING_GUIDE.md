# Video Captioning Guide

NeMo Curator uses Qwen Vision-Language models for video captioning.

## Captioning Pipeline

The captioning workflow consists of three stages:

```
CaptionPreparationStage → CaptionGenerationStage → [CaptionEnhancementStage]
```

### CaptionPreparationStage

Prepares video frames for the caption model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_variant` | `qwen` | Caption model (currently only Qwen) |
| `prompt_variant` | `default` | Prompt template variant |
| `prompt_text` | None | Custom prompt text |
| `sampling_fps` | 2.0 | Frames per second to sample |
| `window_size` | 256 | Number of frames per caption window |
| `remainder_threshold` | 128 | Min frames for final window |
| `preprocess_dtype` | `float16` | Tensor precision |
| `model_does_preprocess` | False | Let model handle preprocessing |

### CaptionGenerationStage

Generates captions using Qwen VL model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_variant` | `qwen` | Model variant |
| `caption_batch_size` | 8 | Batch size for inference |
| `fp8` | False | Use FP8 weights (reduces memory) |
| `max_output_tokens` | 512 | Maximum caption length |
| `disable_mmcache` | True | Disable vLLM multimodal cache |

**GPU Memory Requirements:**
- FP16: ~24GB
- FP8: ~16GB

### CaptionEnhancementStage (Optional)

Refines captions using Qwen LM model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_variant` | `qwen` | Enhancement model |
| `prompt_variant` | `default` | Enhancement prompt |
| `model_batch_size` | 128 | Batch size |
| `fp8` | False | Use FP8 weights |
| `max_output_tokens` | 512 | Maximum output length |

## Prompt Variants

### Default Prompt

General-purpose video description:
```
Describe this video in detail.
```

### AV (Audio-Visual) Prompt

For content with important audio:
```
Describe this video including any audio or spoken content.
```

### AV-Surveillance Prompt

For surveillance-style footage:
```
Describe the events in this video objectively.
```

### Custom Prompt

Use `prompt_text` to provide your own:
```yaml
- _target_: nemo_curator.stages.video.caption.caption_preparation.CaptionPreparationStage
  prompt_text: "Generate a detailed caption focusing on the main subject and actions."
```

## Performance Tuning

### Batch Size

| GPU Memory | Recommended Batch Size |
|------------|------------------------|
| 24GB | 4-8 |
| 40GB | 8-16 |
| 80GB | 16-32 |

### FP8 Quantization

Reduces GPU memory by ~40% with minimal quality loss:

```yaml
- _target_: nemo_curator.stages.video.caption.caption_generation.CaptionGenerationStage
  fp8: true
  caption_batch_size: 12  # Can increase batch size
```

### Sampling Rate

Higher FPS = more context but slower processing:

| sampling_fps | Use Case |
|--------------|----------|
| 1.0 | Fast processing, static scenes |
| 2.0 | Default, balanced |
| 4.0 | Action-heavy content |

## Output Format

Captions are stored in the output metadata:

```json
{
  "clip_id": "video_001_clip_005",
  "caption": "A person walks through a park...",
  "caption_model": "qwen",
  "caption_timestamp": "2025-01-27T12:00:00Z"
}
```

## Example Configurations

### Standard Captioning

```yaml
- _target_: nemo_curator.stages.video.caption.caption_preparation.CaptionPreparationStage
  model_variant: qwen
  sampling_fps: 2.0

- _target_: nemo_curator.stages.video.caption.caption_generation.CaptionGenerationStage
  model_dir: ${model_dir}
  model_variant: qwen
  caption_batch_size: 8
```

### High-Quality with Enhancement

```yaml
- _target_: nemo_curator.stages.video.caption.caption_preparation.CaptionPreparationStage
  model_variant: qwen
  sampling_fps: 4.0

- _target_: nemo_curator.stages.video.caption.caption_generation.CaptionGenerationStage
  model_dir: ${model_dir}
  caption_batch_size: 4
  max_output_tokens: 1024

- _target_: nemo_curator.stages.video.caption.caption_enhancement.CaptionEnhancementStage
  model_dir: ${model_dir}
```

### Memory-Efficient (FP8)

```yaml
- _target_: nemo_curator.stages.video.caption.caption_generation.CaptionGenerationStage
  model_dir: ${model_dir}
  fp8: true
  caption_batch_size: 12
```

## Troubleshooting

### Out of Memory

1. Reduce `caption_batch_size`
2. Enable FP8 quantization
3. Reduce `sampling_fps`
4. Process fewer clips per task

### Slow Processing

1. Increase `caption_batch_size`
2. Reduce `sampling_fps`
3. Reduce `max_output_tokens`
4. Use more GPU workers

### Poor Caption Quality

1. Increase `sampling_fps` for fast-moving content
2. Try different `prompt_variant`
3. Enable `CaptionEnhancementStage`
4. Increase `max_output_tokens`
