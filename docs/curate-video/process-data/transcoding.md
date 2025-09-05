---
description: "Transcode clips with CPU or GPU encoders and tune for quality and performance"
categories: ["video-curation"]
tags: ["transcoding", "encoding", "ffmpeg", "nvenc", "h264"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "howto"
modality: "video-only"
---


(video-process-transcoding)=

# Clip Encoding

Convert extracted clip buffers into compressed media files suitable for storage and training workflows using encoders. NeMo Curator provides both CPU and GPU-based encoders.

## Use Cases

- Convert raw clip buffers into a standard format (such as H.264 in MP4) for portability.
- Normalize heterogeneous inputs (encoding formats, bit rates, containers) into a consistent output.
- Reduce storage footprint with controlled quality settings.

## Before You Start

If you only need embeddings or analysis and do not require saved media files, you can skip encoding. When writing clips, NeMo Curator produces `.mp4` by default.

---

## Quickstart

Use the pipeline stage or the example script flags to encode clips with CPU or GPU encoders.

::::{tab-set}

:::{tab-item} Pipeline Stage

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.video.clipping.clip_extraction_stages import FixedStrideExtractorStage, ClipTranscodingStage

pipe = Pipeline(name="transcode_example")
pipe.add_stage(FixedStrideExtractorStage(clip_len_s=10.0, clip_stride_s=10.0))
pipe.add_stage(ClipTranscodingStage(encoder="libopenh264", encode_batch_size=16, encoder_threads=1, verbose=True))
pipe.run()
```

:::

:::{tab-item} Script Flags

```bash
python -m ray_curator.examples.video.video_split_clip_example \
  ...
  --transcode-encoder h264_nvenc \
  --transcode-use-hwaccel \
```

:::
::::

## Encoder Options

```{list-table} Encoders
:header-rows: 1

* - Encoder
  - Hardware
  - Description
* - `libx264`
  - CPU
  - Widely available, high quality, CPU-based.
* - `libopenh264`
  - CPU
  - Good quality and throughput balance. Often faster than `libx264` at similar presets.
* - `h264_nvenc`
  - NVIDIA GPU (NVENC)
  - Uses NVENC for high-throughput H.264 encoding on NVIDIA GPU hardware.
```

```{tip}
On systems with supported NVIDIA GPU hardware and an `ffmpeg` build with NVENC, `h264_nvenc` can significantly increase throughput. Refer to the verification steps below to confirm NVENC availability.
```

### Verify `ffmpeg`/NVENC Support

To use `h264_nvenc`, confirm that your `ffmpeg` build includes NVENC support and install the GPU drivers:

```bash
ffmpeg -hide_banner -encoders | grep nvenc
ffmpeg -hide_banner -hwaccels | grep -i nv
nvidia-smi
```

Expected output includes entries like `V..... h264_nvenc` and `cuda` in the hardware accelerators list. If not present, install an `ffmpeg` build with NVENC and ensure NVIDIA drivers and CUDA are available.

## Configure

Use `ClipTranscodingStage` to control encoder choice, batching, and acceleration:

```python
from nemo_curator.stages.video.clipping.clip_extraction_stages import ClipTranscodingStage

transcode = ClipTranscodingStage(
    encoder="h264_nvenc",        # or "libopenh264", "libx264"
    use_hwaccel=True,             # enable NVENC when using h264_nvenc
    encoder_threads=1,            # CPU thread count for CPU encoders
    encode_batch_size=16,         # number of clips per encode batch
    num_clips_per_chunk=32,       # chunking for downstream writing
    use_input_bit_rate=False,     # set True to preserve input bit rate when available
    num_cpus_per_worker=6.0,
    verbose=True,
)
```

### Parameters

```{list-table} Common Parameters
:header-rows: 1

* - Parameter
  - Description
* - `encoder`
  - Selects the encoding backend. Recommended defaults: `libopenh264` (CPU) or `h264_nvenc` (GPU).
* - `use_hwaccel`
  - Enable when using GPU encoders like `h264_nvenc`.
* - `encoder_threads`
  - CPU threads per worker for CPU encoders. Increase to use more CPU.
* - `encode_batch_size`
  - Batching size for clips; larger batches can improve throughput.
* - `use_input_bit_rate`
  - If True, attempts to reuse the input bit rate; otherwise, the encoder uses its default rate control.
```

```{seealso}
Refer to the quickstart options in [Get Started with Video Curation](gs-video) for command-line flags `--transcode-encoder` and `--transcode-use-hwaccel`.
```

## Troubleshooting

- "Encoder not found": Your `ffmpeg` build may lack the encoder; verify with `ffmpeg -encoders`.
- "No NVENC capable devices found": Install NVIDIA drivers/CUDA and ensure the GPU is visible in `nvidia-smi`.
- Output mismatch or low quality: Revisit encoder defaults; set explicit bit rate/quality settings as needed, or enable `use_input_bit_rate`.
