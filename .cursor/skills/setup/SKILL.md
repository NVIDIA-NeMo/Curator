---
name: setup
description: |
  Install and configure NeMo Curator with automatic environment detection.
  Detects platform, Docker images, GPU/CUDA, and recommends the best way
  to run NeMo Curator. Use when starting fresh OR returning after a break.
license: Apache-2.0
metadata:
  author: nvidia
  version: "3.1"
  category: setup
  aliases: ["/setup", "install nemo curator", "setup environment", "how do I run"]
allowed-tools: Bash(python:*) Bash(uv:*) Bash(nvidia-smi) Bash(docker:*) Bash(curl) Read
---

# NeMo Curator Setup

Intelligent setup that detects your environment and gets NeMo Curator running.

## Agent Instructions

Follow this workflow. Execute automated steps immediately. For manual steps, clearly tell the user what to do and wait for confirmation before proceeding.

### Step 1: Detect Environment (Automated)

Run the detection script:

```bash
python skills/setup/scripts/detect_environment.py --quick
```

Parse the output to determine the scenario.

### Step 2: Execute Based on Scenario

The script outputs a `scenario` field. Follow the appropriate path:

---

#### Scenario: `native_existing`

**Condition**: Linux + NeMo Curator already installed

**Automated**: Verify installation works:

```bash
python -c "import nemo_curator; print(f'NeMo Curator {nemo_curator.__version__} ready')"
```

**Done**: Tell user they're ready to go.

---

#### Scenario: `native_install`

**Condition**: Linux, NeMo Curator not installed

**Automated**: Install NeMo Curator:

```bash
uv pip install nemo-curator[text_cuda12]
```

Then verify:

```bash
python -c "import nemo_curator; print(f'NeMo Curator {nemo_curator.__version__} installed')"
```

**Done**: Tell user installation complete.

---

#### Scenario: `docker_existing`

**Condition**: Docker running + NeMo Curator image found

**Automated**: Verify image works:

```bash
docker run --rm <IMAGE_NAME> python -c "import nemo_curator; print(f'NeMo Curator {nemo_curator.__version__} ready')"
```

**Done**: Tell user which image to use and provide the run command.

---

#### Scenario: `docker_pull`

**Condition**: Docker running, no NeMo Curator image

**Automated**: Pull the official image:

```bash
docker pull nvcr.io/nvidia/nemo-curator:latest
```

Then verify:

```bash
docker run --rm nvcr.io/nvidia/nemo-curator:latest python -c "import nemo_curator; print('Ready')"
```

**Done**: Tell user the image is ready.

---

#### Scenario: `docker_start`

**Condition**: Docker installed but not running

**⚠️ MANUAL STEP REQUIRED**: The agent cannot start Docker Desktop programmatically.

Tell the user:

> **Action Required**: Start Docker Desktop
>
> 1. Open Docker Desktop from your Applications folder
> 2. Wait for the whale icon in the menu bar to stop animating
> 3. Reply "ready" when Docker is running

**Wait for user confirmation**, then re-run detection:

```bash
python skills/setup/scripts/detect_environment.py --quick
```

Continue with the new scenario (should be `docker_pull` or `docker_existing`).

---

#### Scenario: `docker_install`

**Condition**: Docker not installed

**⚠️ MANUAL STEP REQUIRED**: The agent cannot install Docker.

Tell the user:

> **Action Required**: Install Docker Desktop
>
> 1. Download from: https://www.docker.com/products/docker-desktop/
> 2. Install and start Docker Desktop
> 3. Reply "ready" when Docker is running

**Wait for user confirmation**, then re-run detection.

---

### Step 3: Verify Setup (Automated)

After any successful path, provide:

1. **The run command** for their environment
2. **A test command** they can try

Example output:

```
✅ NeMo Curator is ready!

Run command:
  docker run --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/nemo-curator:latest python your_script.py

Test it:
  docker run --rm nvcr.io/nvidia/nemo-curator:latest python -c "import nemo_curator; print('Works!')"
```

---

## Automation Summary

| Scenario | Automated? | Agent Action |
|----------|------------|--------------|
| `native_existing` | ✅ Yes | Verify and confirm |
| `native_install` | ✅ Yes | Install via uv, verify |
| `docker_existing` | ✅ Yes | Verify and provide command |
| `docker_pull` | ✅ Yes | Pull image, verify |
| `docker_start` | ❌ No | Prompt user, wait, retry |
| `docker_install` | ❌ No | Prompt user, wait, retry |

---

## Quick Reference

### Docker Run Commands

```bash
# Without GPU (macOS/Windows)
docker run -it --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/nemo-curator:latest

# With GPU (Linux)
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/nemo-curator:latest
```

### PyPI Install (Linux Only)

```bash
# Text curation
uv pip install nemo-curator[text_cuda12]

# Video curation
uv pip install --no-build-isolation nemo-curator[video_cuda12]

# Everything
uv pip install --no-build-isolation nemo-curator[all]
```

---

## References

- [references/TEXT_PACKAGES.md](references/TEXT_PACKAGES.md) - Text dependencies
- [references/VIDEO_PACKAGES.md](references/VIDEO_PACKAGES.md) - Video dependencies + FFmpeg
- [references/IMAGE_PACKAGES.md](references/IMAGE_PACKAGES.md) - Image dependencies
- [references/AUDIO_PACKAGES.md](references/AUDIO_PACKAGES.md) - Audio dependencies
- [references/TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) - Common issues and fixes
