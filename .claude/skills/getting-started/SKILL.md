---
description: Set up NeMo Curator for data curation (text, image, video, audio)
---

# Getting Started with NeMo Curator

Guide users through setting up NeMo Curator for their data curation needs. This skill prioritizes using `uv` for fast, reliable dependency management.

## Step 1: Ask Questions

Use `AskUserQuestion` to ask these questions in a single call:

### Q1: Data Modality (multiSelect: true)
**Question:** "What type of data will you be working with?"
- **Text** - LLM training data curation (filtering, deduplication, PII redaction)
- **Images** - Vision model training (embeddings, NSFW filtering, deduplication)
- **Video** - Video processing (splitting, captioning, embeddings)
- **Audio** - Speech data (ASR transcription, quality filtering)

### Q2: Compute Environment
**Question:** "What's your compute environment?"
- **Local (no GPU)** - CPU-only, limited to smaller datasets
- **Local (with NVIDIA GPU)** - GPU-accelerated processing
- **Docker container** - Containerized deployment (recommended for video)
- **Cloud/HPC cluster** - Distributed processing with Ray

### Q3: Synthetic Data
**Question:** "Will you be generating synthetic data with LLMs?"
- **Yes** - Requires NVIDIA API key
- **No / Not sure**

### Q4: Gated Models
**Question:** "Will you use gated HuggingFace models (e.g., Llama tokenizers)?"
- **Yes** - Requires HuggingFace token
- **No / Not sure**

## Step 2: Platform Compatibility Check

**IMPORTANT: Check platform compatibility before proceeding.**

### Platform Support Matrix
| Platform | Text | Image | Video | Audio | GPU Acceleration |
|----------|------|-------|-------|-------|------------------|
| **Linux x86_64** | ✅ | ✅ | ✅ | ✅ | ✅ Full support |
| **Linux ARM64** | ✅ | ✅ | ❌ | ✅ | ⚠️ Limited |
| **macOS (Intel)** | ✅ | ✅ | ❌ | ✅ | ❌ No CUDA |
| **macOS (Apple Silicon)** | ✅ | ⚠️ | ❌ | ✅ | ❌ No CUDA |
| **Windows** | ❌ | ❌ | ❌ | ❌ | ❌ Not supported |

If user selected **Video** on macOS or Windows, warn them:
```
⚠️ Video curation requires Linux x86_64 with NVIDIA GPU.
Consider using Docker or a cloud environment instead.
```

Check the platform:
```bash
uname -s && uname -m
```

## Step 3: Environment Checks

Run these checks in parallel:

### 3.1 Python Version (REQUIRED: 3.10, 3.11, or 3.12)
```bash
python3 --version
```
**Python 3.13+ is NOT supported** due to dependency constraints.

If wrong version, advise:
```
NeMo Curator requires Python 3.10, 3.11, or 3.12.
Install with pyenv or your system package manager.
```

### 3.2 Check for existing virtual environment
```bash
echo $VIRTUAL_ENV
```

### 3.3 Check if uv is installed
```bash
which uv && uv --version
```

### 3.4 GPU Check (if GPU selected)
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

### GPU VRAM Requirements
| Modality | Minimum VRAM | Recommended |
|----------|--------------|-------------|
| Text (GPU) | 16 GB | 24+ GB |
| Image | 24 GB | 48 GB |
| Video | 21 GB (optimized) | 38 GB (full) |
| Audio | 16 GB | 24+ GB |

Warn if GPU VRAM is insufficient for selected modality.

## Step 4: Install uv (PRIORITY)

**uv is the recommended package manager for NeMo Curator.** It provides faster dependency resolution and better handling of complex extras.

If uv is not installed, ask permission and install:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then source it:
```bash
source $HOME/.local/bin/env
```

Verify installation:
```bash
uv --version
```

## Step 5: Create Virtual Environment

If not in a virtual environment, create one with uv:
```bash
uv venv --python 3.11
source .venv/bin/activate
```

Or specify exact Python version if needed:
```bash
uv venv --python 3.10
```

## Step 6: Video Prerequisites (if Video modality selected)

**Video curation has additional requirements:**

### 6.1 FFmpeg with H.264 Encoders
Video processing requires FFmpeg compiled with specific encoders. Check if already installed:
```bash
ffmpeg -encoders 2>/dev/null | grep -E "h264|264"
```

Should show at least one of: `h264_nvenc` (GPU), `libx264`, or `libopenh264`.

If not available, on Ubuntu/Debian:
```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/docker/common/install_ffmpeg.sh -o install_ffmpeg.sh
chmod +x install_ffmpeg.sh
sudo bash install_ffmpeg.sh
```

**Note:** This installs ~30 system packages and builds FFmpeg from source. Takes 10-20 minutes.

**Alternative:** Use the Docker container which has FFmpeg pre-built.

### 6.2 System packages for FFmpeg build
If building FFmpeg, these are required:
```bash
sudo apt-get install -y build-essential autoconf automake cmake pkg-config libtool nasm yasm
```

## Step 7: Install NeMo Curator

### Build the install command based on selections:

**Single modality (CPU):**
```bash
uv pip install "nemo-curator[text_cpu]"
uv pip install "nemo-curator[image_cpu]"
uv pip install "nemo-curator[video_cpu]"
uv pip install "nemo-curator[audio_cpu]"
```

**Single modality (GPU/CUDA 12):**
```bash
uv pip install "nemo-curator[text_cuda12]"
uv pip install "nemo-curator[image_cuda12]"
uv pip install --no-build-isolation "nemo-curator[video_cuda12]"
uv pip install "nemo-curator[audio_cuda12]"
```

**Multiple modalities - combine extras:**
```bash
uv pip install "nemo-curator[text_cuda12,image_cuda12]"
```

**Important flags:**
- `--no-build-isolation` is **required** for `video_cuda12` due to flash-attn compilation requirements

### Version Constraints to Know
- PyTorch is limited to ≤2.9.1 for video modality
- scikit-learn must be <1.8.0 (handled automatically)
- These are managed by uv, but be aware if you have existing installations

### For Docker users:
Recommend using the pre-built NGC container (includes FFmpeg and all dependencies):
```bash
docker pull nvcr.io/nvidia/nemo-curator:25.09
docker run --gpus all -it --rm nvcr.io/nvidia/nemo-curator:25.09
```
This container includes CUDA 12.8.1, Python 3.12, Ubuntu 24.04, and all modalities pre-installed.

## Step 8: API Key Guidance

Based on answers to Q3 and Q4, provide relevant guidance:

### If synthetic data generation (Q3 = Yes):
```
To use synthetic data generation, you'll need an NVIDIA API key.

1. Get your key at: https://build.nvidia.com/settings/api-keys
2. Press CTRL/CMD+Z, then type 'fg' to return to your shell
3. Export the key:
   export NVIDIA_API_KEY="your-key-here"

Or add to a .env file in your project.
```

### If gated HuggingFace models (Q4 = Yes):
```
To use gated models like Llama tokenizers, you'll need a HuggingFace token.

1. Get your token at: https://huggingface.co/settings/tokens
2. Accept the model license on HuggingFace (e.g., for Llama models)
3. Press CTRL/CMD+Z, then type 'fg' to return to your shell
4. Export the token:
   export HF_TOKEN="your-token-here"
```

## Step 9: Verify Installation

Run verification with user permission. Execute the verifications that apply based on the user's selections from Step 1.

### 9.1 Basic Import Check (ALWAYS run)
```bash
python3 -c "import nemo_curator; print(f'NeMo Curator {nemo_curator.__version__} installed')"
```

### 9.2 Ray Cluster Verification (ALWAYS run)
Verify Ray can initialize and stop correctly:
```bash
python3 .claude/skills/getting-started/scripts/verify_ray.py
```

### 9.3 Modality-Specific Import Checks (run based on Q1 selection)

| Modality | Command |
|----------|---------|
| Text | `python3 .claude/skills/getting-started/scripts/verify_text.py` |
| Image | `python3 .claude/skills/getting-started/scripts/verify_image.py` |
| Video | `python3 .claude/skills/getting-started/scripts/verify_video.py` |
| Audio | `python3 .claude/skills/getting-started/scripts/verify_audio.py` |

### 9.4 GPU Verification (if GPU selected in Q2)
```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"
```

### 9.5 Video Encoder Verification (if Video selected in Q1)
```bash
ffmpeg -encoders 2>/dev/null | grep -E "h264_nvenc|libx264|libopenh264" && echo "✓ H.264 encoder available"
```

### 9.6 Pipeline Execution Verification (run based on Q2 selection)

| Environment | Command |
|-------------|---------|
| GPU users | `python tutorials/quickstart.py` |
| CPU-only users | `python3 .claude/skills/getting-started/scripts/verify_pipeline_cpu.py` |

## Step 10: Data Formats & Storage

**Supported input/output formats:**
| Modality | Formats |
|----------|---------|
| Text | JSONL (primary), Parquet |
| Image | WebDataset tar archives |
| Video | Local paths, S3, HTTP(S) URLs |
| Audio | Local files with JSONL manifests |

**Cloud storage (S3):** Configure via `storage_options` dict passed to readers:
```python
storage_options = {"key": "AWS_KEY", "secret": "AWS_SECRET"}
```

## Step 11: Ray & Network (Multi-Node/Cluster)

If running on a cluster or multi-node setup, be aware of Ray ports:

| Port | Service | Purpose |
|------|---------|---------|
| 6379 | Ray GCS | Cluster communication |
| 8265 | Ray Dashboard | Monitoring UI (http://localhost:8265) |
| 10001 | Ray Client | Python client connection |
| 10002-19999 | Workers | Worker communication range |

**Firewall rules:** Ensure these ports are open between head and worker nodes.

**Debugging tip:** Access the Ray dashboard at `http://localhost:8265` to monitor jobs, view logs, and check resource usage.

**Environment variable:** Set `CURATOR_IGNORE_RAY_HEAD_NODE=1` to override Ray head node detection if needed.

**Execution backends:** NeMo Curator uses `XennaExecutor` by default (recommended). Alternative experimental backends exist (`RayActorPoolExecutor`, `RayDataExecutor`) but stick with the default for getting started.

## Step 12: First-Run Notes

**Important:** Some models download on first use:
- **Text classifiers** (quality, domain, safety): 100MB - 2GB each
- **Image CLIP models**: ~1.5GB
- **Video embedding models**: Cosmos-Embed1 (varies by resolution)
- **Audio ASR models**: 100MB - 1GB depending on language

These download to `~/.cache/huggingface/` by default. Ensure you have disk space and network access on first run.

## Step 13: Recommend Tutorial

Based on modality selection, recommend the most relevant tutorial from https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials/:

| Selection | Recommended Tutorial | Description |
|-----------|---------------------|-------------|
| **Text (beginner)** | `tutorials/text/tinystories/` | Small dataset, end-to-end pipeline (~22k samples) |
| **Text (large-scale)** | `tutorials/text/llama-nemotron-data-curation/` | 30M sample pipeline with filtering |
| **Text (PII)** | `tutorials/text/gliner-pii-redaction/` | Privacy/PII redaction with GLiNER |
| **Image** | `tutorials/image/getting-started/image_curation_example.py` | CLIP embeddings, aesthetic scoring, NSFW filtering |
| **Video** | `tutorials/video/getting-started/video_split_clip_example.py` | Splitting, scene detection, captioning |
| **Audio** | `tutorials/audio/fleurs/` | Multilingual ASR with FLEURS dataset |
| **Synthetic** | `tutorials/synthetic/synthetic_data_generation_example.py` | LLM-based Q&A generation |

**Always also recommend:** `tutorials/quickstart.py` for understanding the core pipeline architecture (Tasks, Stages, Pipelines).

**Pipeline config YAMLs:** Pre-built configs exist in `nemo_curator/config/text/` for common workflows:
- `exact_deduplication_pipeline.yaml`
- `semantic_deduplication_pipeline.yaml`
- `fuzzy_deduplication_pipeline.yaml`
- `heuristic_filter_pipeline.yaml`

## Step 14: Summary Output

At the end, provide a summary:
```
NeMo Curator Setup Complete!

Platform: {Linux/macOS} {x86_64/arm64}
Python: {version}
Package manager: uv {version}
Installed: nemo-curator[{extras}]
Modalities: {selected modalities}
Compute: {CPU/GPU with VRAM}
API Keys needed: {list or "None"}

First-run note: Models will download on first use (~1-2GB per modality)

Next steps:
1. {If API keys needed: Set your API keys (CTRL/CMD+Z -> fg)}
2. Try the quickstart: python tutorials/quickstart.py
3. Then try: {recommended tutorial path}
4. Docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemocurator/
```

## Troubleshooting

If installation fails, check:

1. **"No matching distribution"** - Wrong Python version (need 3.10-3.12)
2. **CUDA errors** - Ensure CUDA 12.x drivers installed (`nvidia-smi`)
3. **flash-attn build fails** - Use `--no-build-isolation` flag
4. **scikit-learn conflicts** - NeMo Curator requires <1.8.0
5. **FFmpeg encoder missing** - Run the FFmpeg install script
6. **Out of memory** - Check GPU VRAM requirements above
7. **Ray port conflicts** - Check if ports 6379, 8265, 10001 are in use
8. **Job interrupted** - Resumable: compare input/output file counts to find remaining work

**Debugging:**
- Ray dashboard: `http://localhost:8265`
- Logs use `loguru` - check stdout/stderr for detailed traces
