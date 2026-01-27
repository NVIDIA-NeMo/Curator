---
description: "Release notes and version history for NeMo Curator platform updates and new features"
categories: ["reference"]
tags: ["release-notes", "changelog", "updates"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused", "devops-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(about-release-notes)=

# NeMo Curator Release Notes: {{ current_release }}

This major release represents a fundamental architecture shift from [Dask](https://www.dask.org/) to [Ray](https://www.ray.io/), expanding NeMo Curator to support multimodal data curation with new [video](../../curate-video/index.md) and [audio](../../curate-audio/index.md) capabilities. This refactor enables unified backend processing, better heterogeneous computing support, and enhanced autoscaling for dynamic workloads.

**Migrating from a previous version of NeMo Curator?** Refer to the {ref}`Migration Guide <migration-guide>` for step-by-step instructions and the {ref}`Migration FAQ <migration-faq>` for common questions.

## Installation Updates

- **New Docker container**: Updated Docker infrastructure with CUDA 12.8.1 and Ubuntu 24.04 base; obtainable through the [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator) (`nvcr.io/nvidia/nemo-curator:{{ container_version }}`)
- **Docker file to build own image**: Simplified [Dockerfile](https://github.com/NVIDIA-NeMo/Curator/blob/main/docker/Dockerfile) structure for custom container builds with FFmpeg support
- **UV source installations**: Integrated UV package manager (v0.8.22) for faster dependency management
- **PyPI improvements**: Enhanced PyPI installation with modular extras for targeted functionality:

  ```{list-table} Available Installation Extras
  :header-rows: 1
  :widths: 25 35 40

  * - Extra
    - Installation Command
    - Description
  * - **All Modalities**
    - `nemo-curator[all]`
    - Complete installation with all modalities and GPU support
  * - **Text Curation**
    - `nemo-curator[text_cuda12]`
    - GPU-accelerated text processing with RAPIDS
  * - **Image Curation**
    - `nemo-curator[image_cuda12]`
    - Image processing with NVIDIA DALI
  * - **Audio Curation**
    - `nemo-curator[audio_cuda12]`
    - Speech recognition with NeMo ASR models
  * - **Video Curation**
    - `nemo-curator[video_cuda12]`
    - Video processing with GPU acceleration
  * - **Basic GPU**
    - `nemo-curator[cuda12]`
    - CUDA utilities without modality-specific dependencies
  ```

  All GPU installations require the NVIDIA PyPI index:
  ```bash
  uv pip install https://pypi.nvidia.com nemo-curator[EXTRA]
  ```

## New Modalities

### Video

NeMo Curator now supports comprehensive [video data curation](../../curate-video/index.md) with distributed processing capabilities:

- **Video splitting**: [Fixed-stride](../../curate-video/process-data/clipping.md) and [scene-change detection (TransNetV2)](../../curate-video/process-data/clipping.md) for clip extraction
- **Semantic deduplication**: [K-means clustering and pairwise similarity](../../curate-video/process-data/dedup.md) for near-duplicate clip removal
- **Content filtering**: [Motion-based filtering](../../curate-video/process-data/filtering.md) and [aesthetic filtering](../../curate-video/process-data/filtering.md) for quality improvement
- **Embedding generation**: InternVideo2 and Cosmos-Embed1 models for clip-level embeddings
- **Enhanced captioning**: [VL-based caption generation with optional LLM-based rewriting](../../curate-video/process-data/captions-preview.md) (Qwen-VL and Qwen-LM supported) for detailed video descriptions
- **Ray-based distributed architecture**: Scalable video processing with [autoscaling support](../concepts/video/architecture.md)

### Audio

New [audio curation capabilities](../../curate-audio/index.md) for speech data processing:

- **ASR inference**: [Automatic speech recognition](../../curate-audio/process-data/asr-inference/index.md) using NeMo Framework pretrained models
- **Quality assessment**: [Word Error Rate (WER) and Character Error Rate (CER)](../../curate-audio/process-data/quality-assessment/index.md) calculation
- **Speech metrics**: [Duration analysis and speech rate metrics](../../curate-audio/process-data/audio-analysis/index.md) (words/characters per second)
- **Text integration**: Seamless integration with [text curation workflows](../../curate-audio/process-data/text-integration/index.md) via `AudioToDocumentStage`
- **Manifest support**: JSONL manifest format for audio file management

## Modality Refactors

### Text

- **Ray backend migration**: Complete transition from Dask to Ray for distributed [text processing](../../curate-text/index.md)
- **Improved model-based classifier throughput**: Better overlapping of compute between tokenization and inference through [length-based sequence sorting](../../curate-text/process-data/quality-assessment/distributed-classifier.md) for optimal GPU memory utilization
- **Task-centric architecture**: New `Task`-based processing model for finer-grained control
- **Pipeline redesign**: Updated `ProcessingStage` and `Pipeline` architecture with resource specification

### Image

- **Pipeline-based architecture**: Transitioned from legacy `ImageTextPairDataset` to modern [stage-based processing](../../curate-images/index.md) with `ImageReaderStage`, `ImageEmbeddingStage`, and filter stages
- **DALI-based image loading**: New `ImageReaderStage` uses NVIDIA DALI for high-performance WebDataset tar shard processing with GPU/CPU fallback
- **Modular processing stages**: Separate stages for [embedding generation](../../curate-images/process-data/embeddings/index.md), [aesthetic filtering](../../curate-images/process-data/filters/aesthetic.md), and [NSFW filtering](../../curate-images/process-data/filters/nsfw.md)
- **Task-based data flow**: Images processed as `ImageBatch` tasks containing `ImageObject` instances with metadata, embeddings, and classification scores

Learn more about [image curation](../../curate-images/index.md).

## Deduplication Improvements

Enhanced deduplication capabilities across all modalities with improved performance and flexibility:

- **Exact and Fuzzy deduplication**: Updated [rapidsmpf-based shuffle backend](../../reference/infrastructure/gpu-processing.md) for more efficient GPU-to-GPU data transfer and better spilling capabilities
- **Semantic deduplication**: Support for deduplicating [text](../../curate-text/process-data/deduplication/semdedup.md) and [video](../../curate-video/process-data/dedup.md) datasets using unified embedding-based workflows
- **New ranking strategies**: Added `RankingStrategy` which allows you to rank elements within cluster centers to decide which point to prioritize during duplicate removal, supporting [metadata-based ranking](../../curate-text/process-data/deduplication/semdedup.md) to prioritize specific datasets or inputs

## Core Refactors

The architecture refactor introduces a layered system with unified interfaces and multiple execution backends:

```{mermaid}
graph LR
    subgraph "User Layer"
        P[Pipeline]
        S1[ProcessingStage X→Y]
        S2[ProcessingStage Y→Z]
        S3[ProcessingStage Z→W]
        R[Resources<br/>CPU/GPU/NVDEC/NVENC]
    end
    
    subgraph "Orchestration Layer"
        BE[BaseExecutor Interface]
    end
    
    subgraph "Backend Layer"
        XE[XennaExecutor]
        RAP[RayActorPoolExecutor]
        RDE[RayDataExecutor]
    end
    
    subgraph "Adaptation Layer"
        XA[Xenna Adapter]
        RAPA[Ray Actor Adapter]
        RDA[Ray Data Adapter]
    end
    
    subgraph "Execution Layer"
        X[Cosmos-Xenna<br/>Streaming/Batch]
        RAY1[Ray Actor Pool<br/>Load Balancing]
        RAY2[Ray Data API<br/>Dataset Processing]
    end
    
    P --> S1
    P --> S2
    P --> S3
    S1 -.-> R
    S2 -.-> R
    S3 -.-> R
    
    P --> BE
    BE --> XE
    BE --> RAP
    BE --> RDE
    
    XE --> XA
    RAP --> RAPA
    RDE --> RDA
    
    XA --> X
    RAPA --> RAY1
    RDA --> RAY2
    
    style P fill:#E6F3FF
    style BE fill:#F0F8FF
```

### Pipelines

- **New Pipeline API**: Ray-based pipeline execution with `BaseExecutor` interface
- **Multiple backends**: Support for [Xenna, Ray Actor Pool, and Ray Data execution backends](../../reference/infrastructure/execution-backends.md)
- **Resource specification**: Configurable CPU and GPU memory requirements per stage
- **Stage composition**: Improved stage validation and execution orchestration

### Stages

- **ProcessingStage redesign**: Generic `ProcessingStage[X, Y]` base class with type safety
- **Resource requirements**: Built-in resource specification for CPU and GPU memory
- **Backend adapters**: Stage adaptation layer for different Ray orchestration systems
- **Input/output validation**: Enhanced type checking and data validation

## Tutorials

- **Text tutorials**: Updated all [text curation tutorials](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/text) to use new Ray-based API
- **Image tutorials**: Migrated [image processing tutorials](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/image) to unified backend
- **Audio tutorials**: New [audio curation tutorials](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/audio)
- **Video tutorials**: New [video processing tutorials](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/video)

For all tutorial content, refer to the [tutorials directory](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials) in the NeMo Curator GitHub repository.

## Known Limitations

> (Pending Refactor in Future Release)

### Generation

- **Synthetic data generation**: Synthetic text generation features are being refactored for Ray compatibility
- **Hard negative mining**: Retrieval-based data generation workflows under development

### PII

- **PII processing**: Personal Identifiable Information removal tools are being updated for Ray backend
- **Privacy workflows**: Enhanced privacy-preserving data curation capabilities in development

### Blending & Shuffling

- **Data blending**: Multi-source dataset blending functionality being refactored
- **Dataset shuffling**: Large-scale data shuffling operations under development

## Docs Refactor

- **Local preview capability**: Improved documentation build system with local preview support
- **Modality-specific guides**: Comprehensive documentation for each supported modality ([text](../../curate-text/index.md), [image](../../curate-images/index.md), [audio](../../curate-audio/index.md), [video](../../curate-video/index.md))
- **API reference**: Complete [API documentation](../../apidocs/index.rst) with type annotations and examples

---

## What's Next

The next release will focus on completing the refactor of Synthetic Data Generation, PII, and Blending & Shuffling features, along with additional performance optimizations and new modality support.

```{toctree}
:hidden:
:maxdepth: 4

Migration Guide <migration-guide>
Migration FAQ <migration-faq>

```
