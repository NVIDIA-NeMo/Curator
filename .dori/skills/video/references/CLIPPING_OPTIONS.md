# Video Clipping Options

NeMo Curator provides two methods for splitting videos into clips.

## TransNetV2 (ML-based Scene Detection)

Uses a neural network to detect scene boundaries based on visual transitions.

### When to Use

- Movies, TV shows, and edited video content
- Content with clear scene cuts
- When you need semantically meaningful clips
- Higher quality training data

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `threshold` | 0.4 | 0.0-1.0 | Scene detection sensitivity |
| `min_length_s` | 2.0 | 0.5-30 | Minimum clip length in seconds |
| `max_length_s` | 10.0 | 2-60 | Maximum clip length in seconds |
| `gpu_memory_gb` | 16.0 | 8-24 | GPU memory allocation |

### GPU Requirements

- **Default**: 10GB GPU memory (configurable via `gpu_memory_gb` parameter)
- **Note**: VideoFrameExtractionStage (runs before TransNetV2) also requires 10GB with pynvc decoder

## Fixed Stride (Duration-based)

Splits videos at fixed time intervals regardless of content.

### When to Use

- Continuous content (lectures, surveillance)
- Consistent clip lengths required
- Lower compute requirements

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_len_s` | 10.0 | Duration of each clip in seconds |
| `clip_stride_s` | 10.0 | Interval between clip starts |
| `min_clip_length_s` | 2.0 | Minimum length for final clip |

### GPU Requirements

- **None**: Runs entirely on CPU

## Comparison

| Feature | TransNetV2 | Fixed Stride |
|---------|------------|--------------|
| GPU Required | Yes (10GB default) | No |
| Processing Speed | ~1 video/min | ~10 videos/min |
| Clip Boundaries | Scene-based | Time-based |
| Best For | Edited content | Continuous content |
