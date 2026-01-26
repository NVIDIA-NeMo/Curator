---
name: setup
description: |
  Install and configure NeMo Curator with automatic environment detection.
  Detects GPU/CUDA, chooses appropriate packages, executes installation,
  and verifies success. Use when starting with NeMo Curator or troubleshooting
  installation issues.
license: Apache-2.0
metadata:
  author: nvidia
  version: "2.0"
  category: setup
  aliases: ["/setup", "install nemo curator", "setup environment"]
allowed-tools: Bash(python:*) Bash(uv:*) Bash(nvidia-smi) Bash(curl) Read
---

# NeMo Curator Setup

Install NeMo Curator with automatic environment detection.

## Workflow

### Step 1: Detect Environment

Run environment detection to determine CUDA version, GPU memory, and existing packages:

```bash
python scripts/detect_environment.py
```

Parse the JSON output to get:
- `cuda_version`: CUDA version or null (determines `_cpu` vs `_cuda12`)
- `gpu_memory_gb`: Available VRAM (determines if GPU stages are viable)
- `ffmpeg_installed`: Required for video processing
- `recommended_extras`: Suggested install targets
- `warnings`: Issues to address

### Step 2: Determine Modality

If user specified a modality, use it. Otherwise, ask:

```markdown
What data will you curate?

1. **Text** - JSONL, Parquet, documents (filtering, deduplication, classification)
2. **Video** - MP4, clips (scene detection, captioning, embeddings)
3. **Image** - PNG, JPEG (CLIP embeddings, aesthetic filtering)
4. **Audio** - WAV, speech (ASR transcription, WER)
5. **All** - Everything (largest install)
```

### Step 3: Select Install Target

Based on environment detection + modality:

| Modality | CUDA 12.x | No CUDA | Special Flags |
|----------|-----------|---------|---------------|
| text | `text_cuda12` | `text_cpu` | — |
| video | `video_cuda12` | `video_cpu` | `--no-build-isolation` |
| image | `image_cuda12` | `image_cpu` | — |
| audio | `audio_cuda12` | `audio_cpu` | `--override override.txt` |
| all | `all` | — | `--no-build-isolation --override override.txt` |

### Step 4: Handle Prerequisites

#### Video requires FFmpeg

If modality includes video and `ffmpeg_installed` is false:

```markdown
Video processing requires FFmpeg. Install it?

1. **Yes** - Run install script (requires sudo)
2. **Skip** - Continue without GPU encoding support
```

If yes:
```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/docker/common/install_ffmpeg.sh -o /tmp/install_ffmpeg.sh
sudo bash /tmp/install_ffmpeg.sh
```

#### Audio requires transformers override

If modality includes audio:
```bash
echo "transformers==4.55.2" > /tmp/override.txt
```

### Step 5: Execute Installation

```bash
# Pre-requisites
uv pip install torch wheel_stub psutil setuptools

# Install NeMo Curator
uv pip install {flags} nemo-curator[{extras}] {override}
```

Where:
- `{flags}`: `--no-build-isolation` if video_cuda12
- `{extras}`: The selected extras (e.g., `text_cuda12`)
- `{override}`: `--override /tmp/override.txt` if audio

### Step 6: Verify Installation

```bash
python scripts/verify_installation.py --{modality} --json
```

Parse output. If any checks failed:
1. Identify the failed component
2. Check [references/TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) for solution
3. Apply fix
4. Re-run verification

### Step 7: Report Results

```markdown
## ✅ Setup Complete

**Installed**: nemo-curator[text_cuda12]
**Environment**: CUDA 12.4, Python 3.11, 24GB VRAM

### Verification Results
- ✓ NeMo Curator version: 0.5.0
- ✓ Core modules available
- ✓ Text filters (25+)
- ✓ GPU deduplication (cuDF)
- ⚠ vLLM not installed (optional for embeddings)

### Next Steps
1. Start curating: `/curate`
2. Explore filters: `/filter`
3. See stages: `/stages`
```

If failures occurred:

```markdown
## ⚠ Setup Incomplete

**Installed**: nemo-curator[video_cuda12]
**Issues**: 2 components failed verification

### Failed Components
- ✗ flash-attn: Build failed
- ✗ FFmpeg: Not installed

### Recommended Actions
1. **flash-attn**: `uv pip install --no-build-isolation flash-attn<=2.8.3`
2. **FFmpeg**: Run `/setup` again and select "Yes" for FFmpeg installation

### What Still Works
- Video reading (PyAV) ✓
- Basic video processing ✓
- Scene detection requires flash-attn
```

---

## Quick Install (Skip Workflow)

For users who know what they want:

### PyPI with uv

```bash
# Text curation (GPU)
uv pip install torch wheel_stub psutil setuptools
uv pip install nemo-curator[text_cuda12]

# Video curation (GPU)
uv pip install torch wheel_stub psutil setuptools
uv pip install --no-build-isolation nemo-curator[video_cuda12]

# Everything
echo "transformers==4.55.2" > override.txt
uv pip install --no-build-isolation nemo-curator[all] --override override.txt
```

### Docker

```bash
docker pull nvcr.io/nvidia/nemo-curator:latest
docker run --gpus all -it --rm nvcr.io/nvidia/nemo-curator:latest
```

### Source

```bash
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras --all-groups
```

---

## Decision Points

### GPU Not Detected

If `cuda_version` is null:

```markdown
No CUDA GPU detected. Options:

1. **Continue with CPU** - Install `{modality}_cpu` (slower, no GPU deduplication)
2. **Check GPU setup** - Run `nvidia-smi` to diagnose
3. **Use Docker** - Pre-configured GPU environment
```

### Insufficient GPU Memory

If `gpu_memory_gb` < 16:

```markdown
GPU has {gpu_memory_gb} GB VRAM. Some stages need 16GB+.

Options:
1. **Continue** - Use stages that fit in memory
2. **Reduce batch sizes** - Configure stages with smaller batches
3. **Use CPU for heavy stages** - Mixed CPU/GPU pipeline
```

### Build Failures

If installation fails, check error and apply fix from troubleshooting:

| Error Pattern | Fix |
|---------------|-----|
| `flash_attn` build error | `--no-build-isolation` |
| `cudf` import error | Check CUDA version matches |
| `transformers` conflict | Use override file |

Then retry installation.

---

## References

For detailed package information:
- [references/TEXT_PACKAGES.md](references/TEXT_PACKAGES.md) - Text dependencies
- [references/VIDEO_PACKAGES.md](references/VIDEO_PACKAGES.md) - Video dependencies + FFmpeg
- [references/IMAGE_PACKAGES.md](references/IMAGE_PACKAGES.md) - Image dependencies
- [references/AUDIO_PACKAGES.md](references/AUDIO_PACKAGES.md) - Audio dependencies
- [references/TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) - Common issues and fixes
