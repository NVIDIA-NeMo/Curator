---
name: setup
description: |
  Install and configure NeMo Curator with automatic environment detection.
  Detects platform, Docker images, GPU/CUDA, and recommends the best way
  to run NeMo Curator. Use when starting fresh OR returning after a break.
license: Apache-2.0
metadata:
  author: nvidia
  version: "3.0"
  category: setup
  aliases: ["/setup", "install nemo curator", "setup environment", "how do I run"]
allowed-tools: Bash(python:*) Bash(uv:*) Bash(nvidia-smi) Bash(docker:*) Bash(curl) Read
---

# NeMo Curator Setup

Intelligent setup that detects your environment and recommends the best path.

## Quick Start (Run This First!)

```bash
python scripts/detect_environment.py --quick
```

This will:
1. Check your platform (Linux/macOS/Windows)
2. Find existing Docker images
3. Check for GPU/CUDA
4. **Recommend the best way to run NeMo Curator**

### Example Output

```
============================================================
🎯 RECOMMENDATION
============================================================

Use your existing Docker image: nemo-curator-local:latest

Command:
  docker run --rm -v $(pwd):/workspace -w /workspace nemo-curator-local:latest python your_script.py

⚠️  Warnings:
   - Platform 'darwin' not supported. Use Docker.
```

## Scenarios

The script detects one of these scenarios:

| Scenario | Detection | Recommendation |
|----------|-----------|----------------|
| `native_existing` | Linux + NeMo Curator installed | Run directly |
| `native_install` | Linux, no install | `uv pip install nemo-curator[...]` |
| `docker_existing` | Non-Linux + Docker image found | Use existing image |
| `docker_pull` | Non-Linux + Docker running, no image | `docker pull nvcr.io/nvidia/nemo-curator` |
| `docker_start` | Docker installed but not running | Start Docker Desktop |
| `docker_install` | No Docker | Install Docker Desktop |

## Full Workflow

### Step 1: Detect Environment

```bash
python scripts/detect_environment.py
```

This outputs:
- Platform support status
- Python version
- GPU/CUDA info
- FFmpeg status
- **Docker images found** (with age and size)
- **Recommended scenario and command**

### Step 2: Follow the Recommendation

The script tells you exactly what to do. Common paths:

#### Path A: You Have a Docker Image

```bash
# Use your existing image
docker run --rm -v $(pwd):/workspace -w /workspace nemo-curator-local:latest python your_script.py
```

#### Path B: Docker Running, No Image

```bash
# Pull official image
docker pull nvcr.io/nvidia/nemo-curator:latest

# Then run
docker run --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/nemo-curator:latest python your_script.py
```

#### Path C: On Linux

```bash
# Install directly
uv pip install nemo-curator[text_cuda12]

# Run
python your_script.py
```


## Image Age Warnings

The detection script warns if images are old:

| Age | Status | Action |
|-----|--------|--------|
| < 30 days | ✅ Recent | Good to use |
| 30-90 days | ⚠️ Aging | Consider updating |
| > 90 days | ⚠️ Old | Recommend: `docker pull ...` |

## Quick Install (Skip Detection)

For users who know what they want:

### Docker (Recommended for macOS/Windows)

```bash
# Pull official image
docker pull nvcr.io/nvidia/nemo-curator:latest

# Run with GPU
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/nemo-curator:latest

# Run without GPU
docker run -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/nemo-curator:latest
```

### PyPI with uv (Linux Only)

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

### Source

```bash
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras --all-groups
```

---

## Troubleshooting

### "Platform not supported"

NeMo Curator only runs on Linux. Use Docker on macOS/Windows.

### "Docker not running"

Start Docker Desktop from your Applications folder.

### "No images found"

```bash
docker pull nvcr.io/nvidia/nemo-curator:latest
```

### "Image is old"

```bash
docker pull nvcr.io/nvidia/nemo-curator:latest
```

---

## References

- [references/TEXT_PACKAGES.md](references/TEXT_PACKAGES.md) - Text dependencies
- [references/VIDEO_PACKAGES.md](references/VIDEO_PACKAGES.md) - Video dependencies + FFmpeg
- [references/IMAGE_PACKAGES.md](references/IMAGE_PACKAGES.md) - Image dependencies
- [references/AUDIO_PACKAGES.md](references/AUDIO_PACKAGES.md) - Audio dependencies
- [references/TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) - Common issues and fixes
