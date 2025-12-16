# Comprehensive Dependency Testing Design Document

**Author:** Pablo Garay  
**Date:** December 2025  
**Status:** Proposed

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Current State](#current-state)
4. [Proposed Solution](#proposed-solution)
5. [Implementation Details](#implementation-details)
6. [Test Coverage Matrix](#test-coverage-matrix)
7. [Validation & Success Criteria](#validation--success-criteria)

---

## Executive Summary

This document proposes expanding CI to validate that all NeMo Curator extras and dependency combinations install correctly across supported Python versions. This catches installation issues in CI rather than when users discover them.

**Goal:** Implement/Test the resolution/installation of all dependencies and extras.

---

## Problem Statement

NeMo Curator has **12 optional extras** but CI only tests basic installation:

| Risk | Impact |
|------|--------|
| Broken extras discovered by users | Poor user experience, support burden |
| Dependency conflicts not caught | Releases ship with broken installs |
| Python version incompatibilities | Users on specific versions hit issues |
| Lock file drift from pip behavior | Different results for different install methods |

---

## Current State

### What's Tested (`install-test.yml`)

```yaml
# ✅ Lock file with combined CPU extras
uv sync --locked --extra audio_cpu --extra text_cpu --extra video_cpu

# ✅ Basic pip install (no extras)
pip install .
```

### What's NOT Tested

| Category | Extras | Status |
|----------|--------|--------|
| Individual CPU extras | `audio_cpu`, `image_cpu`, `text_cpu`, `video_cpu` | ❌ Not isolated |
| CUDA extras | `cuda12`, `*_cuda12` | ❌ Not tested |
| Deduplication | `deduplication_cuda12` | ❌ Not tested |
| Full install | `all` | ❌ Not tested |
| Pip with extras | `pip install .[text_cpu]` | ❌ Not tested |

### Available Extras (from `pyproject.toml`)

```toml
[project.optional-dependencies]
# GPU support
cuda12 = ["gpustat", "nvidia-ml-py"]

# Deduplication (GPU)
deduplication_cuda12 = ["cudf-cu12", "cuml-cu12", "pylibcugraph-cu12", ...]

# Audio
audio_cpu = ["nemo_toolkit[asr]==2.4.0"]
audio_cuda12 = ["nemo_curator[audio_cpu]", "nemo_curator[cuda12]"]

# Image
image_cpu = ["torchvision"]
image_cuda12 = ["nemo_curator[image_cpu]", "nvidia-dali-cuda120", ...]

# Text
text_cpu = ["beautifulsoup4", "fasttext", "trafilatura", ...]
text_cuda12 = ["nemo_curator[text_cpu]", "nemo_curator[deduplication_cuda12]"]

# Video
video_cpu = ["av", "opencv-python", "torchvision", "einops", ...]
video_cuda12 = ["nemo_curator[video_cpu]", "cvcuda_cu12", "vllm", ...]

# Everything
all = ["nemo_curator[audio_cuda12]", "nemo_curator[image_cuda12]", ...]
```

---

## Proposed Solution

### Overview

Add three new test jobs to `install-test.yml`:

1. **Individual CPU extras** - Test each CPU extra in isolation
2. **Pip extras** - Test pip installation with extras (user experience)
3. **CUDA extras** - Test GPU extras on self-hosted runner

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        install-test.yml                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  EXISTING JOBS                                                           │
│  ┌────────────────────────┐  ┌────────────────────────┐                 │
│  │ uv-lock-file-test      │  │ pip-install-test       │                 │
│  │ Python 3.10, 3.12      │  │ Python 3.10, 3.12      │                 │
│  │ Combined CPU extras    │  │ Base package only      │                 │
│  └────────────────────────┘  └────────────────────────┘                 │
│                                                                          │
│  NEW JOBS                                                                │
│  ┌────────────────────────┐  ┌────────────────────────┐                 │
│  │ extras-isolation-test  │  │ pip-extras-test        │                 │
│  │ Python 3.10, 3.12      │  │ Python 3.10, 3.12      │                 │
│  │ × 4 CPU extras         │  │ × 4 CPU extras         │                 │
│  │ = 8 jobs               │  │ = 8 jobs               │                 │
│  └────────────────────────┘  └────────────────────────┘                 │
│                                                                          │
│  ┌────────────────────────┐                                             │
│  │ extras-cuda-test       │  (on self-hosted GPU runner)                │
│  │ cuda12, dedup, all     │                                             │
│  │ = 3 jobs               │                                             │
│  └────────────────────────┘                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Job 1: Individual CPU Extras (Lock File)

Tests each CPU extra in isolation using `uv sync --locked`:

```yaml
extras-isolation-test:
  needs: [pre-flight]
  if: !(needs.pre-flight.outputs.docs_only == 'true')
  runs-on: ubuntu-latest
  name: UV Extra (${{ matrix.extra }}) - Py${{ matrix.python-version }}
  strategy:
    fail-fast: false
    matrix:
      python-version: ["3.10", "3.12"]
      extra: [audio_cpu, image_cpu, text_cpu, video_cpu]
  steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive

    - uses: astral-sh/setup-uv@v6
      with:
        python-version: ${{ matrix.python-version }}

    - name: Test lock file with extra
      run: |
        echo "🧪 Testing: uv sync --locked --extra ${{ matrix.extra }}"
        uv sync --link-mode copy --locked --extra ${{ matrix.extra }}
        
        # Verify base import
        uv run python -c "from nemo_curator import package_info; print(f'✅ v{package_info.__version__}')"

    - name: Smoke test extra
      run: |
        case "${{ matrix.extra }}" in
          audio_cpu)
            uv run python -c "from nemo.collections.asr.models import ASRModel; print('✅ ASR importable')"
            ;;
          image_cpu)
            uv run python -c "import torchvision; print('✅ torchvision importable')"
            ;;
          text_cpu)
            uv run python -c "import fasttext; print('✅ fasttext importable')"
            uv run python -c "import trafilatura; print('✅ trafilatura importable')"
            ;;
          video_cpu)
            uv run python -c "import av; print('✅ av (PyAV) importable')"
            uv run python -c "import cv2; print('✅ opencv importable')"
            ;;
        esac
```

### Job 2: Pip with Extras (User Experience)

Tests installation via pip with extras - this is what users do:

```yaml
pip-extras-test:
  needs: [pre-flight]
  if: !(needs.pre-flight.outputs.docs_only == 'true')
  runs-on: ubuntu-latest
  name: Pip Extra (${{ matrix.extra }}) - Py${{ matrix.python-version }}
  strategy:
    fail-fast: false
    matrix:
      python-version: ["3.10", "3.12"]
      extra: [audio_cpu, image_cpu, text_cpu, video_cpu]
  steps:
    - uses: actions/checkout@v4

    - uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install via pip with extra
      run: |
        echo "🧪 Testing: pip install .[${{ matrix.extra }}]"
        python -m pip install --upgrade pip
        pip install --no-cache-dir ".[${{ matrix.extra }}]"
        
        # Verify installation
        python -c "from nemo_curator import package_info; print(f'✅ v{package_info.__version__}')"
        pip show nemo-curator

    - name: Smoke test extra
      run: |
        case "${{ matrix.extra }}" in
          audio_cpu)
            python -c "from nemo.collections.asr.models import ASRModel; print('✅ ASR importable')"
            ;;
          image_cpu)
            python -c "import torchvision; print('✅ torchvision importable')"
            ;;
          text_cpu)
            python -c "import fasttext; print('✅ fasttext importable')"
            ;;
          video_cpu)
            python -c "import av; print('✅ av importable')"
            ;;
        esac
```

### Job 3: CUDA Extras (GPU Runner)

Tests GPU-dependent extras on self-hosted runner:

```yaml
extras-cuda-test:
  needs: [pre-flight]
  if: !(needs.pre-flight.outputs.docs_only == 'true')
  runs-on: self-hosted-nemo
  name: CUDA Extra (${{ matrix.extra }})
  strategy:
    fail-fast: false
    matrix:
      extra: [cuda12, deduplication_cuda12, all]
  steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive

    - name: Test CUDA extra in container
      run: |
        echo "🧪 Testing CUDA extra: ${{ matrix.extra }}"
        
        # Build test container with specific extra
        docker build -f docker/Dockerfile -t test-${{ matrix.extra }} .
        
        # Verify CUDA is available
        docker run --rm --gpus all test-${{ matrix.extra }} \
          python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ CUDA available, device: {torch.cuda.get_device_name(0)}')"

    - name: Smoke test CUDA-specific imports
      run: |
        case "${{ matrix.extra }}" in
          cuda12)
            docker run --rm --gpus all test-${{ matrix.extra }} \
              python -c "import gpustat; print('✅ gpustat importable')"
            ;;
          deduplication_cuda12)
            docker run --rm --gpus all test-${{ matrix.extra }} \
              python -c "import cudf; print('✅ cudf importable')"
            docker run --rm --gpus all test-${{ matrix.extra }} \
              python -c "import cuml; print('✅ cuml importable')"
            ;;
          all)
            docker run --rm --gpus all test-${{ matrix.extra }} \
              python -c "import cudf; import cuml; import torchvision; print('✅ all extras importable')"
            ;;
        esac
```

---

## Test Coverage Matrix

### CPU Extras (ubuntu-latest)

| Extra | UV Lock 3.10 | UV Lock 3.12 | Pip 3.10 | Pip 3.12 | Smoke Test |
|-------|--------------|--------------|----------|----------|------------|
| `audio_cpu` | ✅ | ✅ | ✅ | ✅ | ASRModel import |
| `image_cpu` | ✅ | ✅ | ✅ | ✅ | torchvision import |
| `text_cpu` | ✅ | ✅ | ✅ | ✅ | fasttext, trafilatura |
| `video_cpu` | ✅ | ✅ | ✅ | ✅ | av, opencv import |

### CUDA Extras (self-hosted-nemo)

| Extra | Container Build | CUDA Check | Smoke Test |
|-------|-----------------|------------|------------|
| `cuda12` | ✅ | ✅ | gpustat import |
| `deduplication_cuda12` | ✅ | ✅ | cudf, cuml import |
| `all` | ✅ | ✅ | Combined imports |

### Total New Jobs

| Job Type | Count | Runner |
|----------|-------|--------|
| UV Lock CPU extras | 8 | ubuntu-latest |
| Pip CPU extras | 8 | ubuntu-latest |
| CUDA extras | 3 | self-hosted-nemo |
| **Total** | **19** | |

### Estimated Runtime

| Job | Estimated Time |
|-----|----------------|
| UV Lock CPU (each) | ~3-5 min |
| Pip CPU (each) | ~5-8 min |
| CUDA (each) | ~10-15 min |
| **Total parallel time** | ~15 min (jobs run in parallel) |

---

## Validation & Success Criteria

### Success Criteria

| Criteria | Validation |
|----------|------------|
| All extras install without errors | Exit code 0 |
| Core imports succeed | `from nemo_curator import package_info` |
| Extra-specific imports succeed | Smoke tests pass |
| No dependency conflicts | pip/uv don't report conflicts |
| Works on Python 3.10 and 3.12 | Matrix covers both |

### Failure Scenarios to Catch

| Scenario | How It's Caught |
|----------|-----------------|
| Missing dependency in extra | Install step fails |
| Version conflict | pip/uv resolver fails |
| Import error after install | Smoke test fails |
| Python version incompatibility | Matrix job for that version fails |
| Lock file out of sync | `--locked` flag fails if mismatch |

### Rollout Plan

1. **Phase 1:** Add CPU extras tests (low risk, no GPU cost)
2. **Phase 2:** Add pip extras tests (validates user experience)
3. **Phase 3:** Add CUDA extras tests (requires GPU runner time)

---

## Appendix

### A. Related Files

| File | Purpose |
|------|---------|
| `.github/workflows/install-test.yml` | Installation test workflow |
| `pyproject.toml` | Package dependencies and extras |
| `uv.lock` | Locked dependency versions |
| `docker/Dockerfile` | Container with all extras |

### B. References

- [UV Documentation](https://docs.astral.sh/uv/)
- [pip extras_require](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html)
- [GitHub Actions Matrix](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs)
