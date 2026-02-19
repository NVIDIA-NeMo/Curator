# Video Curation Packages

Reference for video-specific NeMo Curator dependencies.

## Package Extras

| Extra | Use Case |
|-------|----------|
| `video_cpu` | Video processing without GPU encoding |
| `video_cuda12` | Full video curation with GPU encoding |

**Note**: `video_cuda12` requires `--no-build-isolation` flag.

## Dependencies

### video_cpu

| Package | Purpose |
|---------|---------|
| `av` | Video I/O (PyAV) |
| `opencv-python` | Image/video processing |
| `torchvision` | Video transforms |
| `einops` | Tensor operations |
| `easydict` | Configuration |

### video_cuda12 (additional)

| Package | Purpose |
|---------|---------|
| `cvcuda_cu12` | NVIDIA CV-CUDA |
| `flash-attn` | Flash Attention (<=2.8.3) |
| `pycuda` | CUDA bindings |
| `PyNvVideoCodec` | NVIDIA Video Codec SDK |
| `torchaudio` | Audio processing |
| `vllm` | Qwen VL captioning |

## FFmpeg Requirement

FFmpeg is required for video decoding and encoding.

### Install with NVENC Support (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/docker/common/install_ffmpeg.sh -o install_ffmpeg.sh
sudo bash install_ffmpeg.sh
```

Installs FFmpeg with:
- `libopenh264` encoder
- NVIDIA NVENC support
- CUDA acceleration

### Verify FFmpeg

```bash
# Check version
ffmpeg -hide_banner -version | head -n 5

# Check encoders (should show h264_nvenc)
ffmpeg -encoders | grep -E "h264_nvenc|libopenh264|libx264"
```

## GPU Requirements

| Stage | GPU Memory | Notes |
|-------|------------|-------|
| TransNetV2 (scene detection) | ~16 GB | Shared |
| CaptionGeneration (Qwen VL) | ~24 GB | Full GPU |
| CosmosEmbed1 | ~16 GB | Shared |
| InternVideo2 | ~16 GB | Shared |
| NVENC encoding | ~2 GB | Hardware encoder |

**Minimum**: 16GB VRAM for basic workflows
**Recommended**: 24GB+ VRAM for captioning

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| MPEG-4 | `.mp4` | Most common |
| AVI | `.avi` | Legacy |
| QuickTime | `.mov` | Apple |
| WebM | `.webm` | Web-optimized |

## InternVideo2 Setup (Optional)

For InternVideo2 embeddings:

```bash
git clone https://github.com/OpenGVLab/InternVideo.git
cd InternVideo
git checkout 09d872e5093296c6f36b8b3a91fc511b76433bf7

curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Curator/main/external/intern_video2_multimodal.patch -o intern_video2_multimodal.patch
patch -p1 < intern_video2_multimodal.patch
cd ..

uv add InternVideo/InternVideo2/multi_modality
```

## Common Issues

### flash-attn Build Failure

```bash
sudo apt-get install build-essential ninja-build
uv pip install --no-build-isolation flash-attn<=2.8.3
```

### PyNvVideoCodec Issues

Requires NVIDIA Video Codec SDK:

```bash
# Check NVIDIA driver supports NVENC
nvidia-smi

# May need SDK: https://developer.nvidia.com/nvidia-video-codec-sdk
```

### Out of GPU Memory

Use CPU clipping instead of TransNetV2:

```yaml
- _target_: nemo_curator.stages.video.clipping.clip_extraction_stages.FixedStrideExtractorStage
  clip_duration: 10.0
```
