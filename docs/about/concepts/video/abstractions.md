---
description: "Key abstractions in video curation including stages, pipelines, and execution modes for scalable processing"
categories: ["concepts-architecture"]
tags: ["abstractions", "pipeline", "stages", "video-curation", "distributed", "ray"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "video-only"
only: not ga
---

(about-concepts-video-abstractions)=

# Key Abstractions

NeMo Curator introduces core abstractions to organize and scale video curation workflows:

- **Pipelines**: Ordered sequences of stages forming an end-to-end workflow.
- **Stages**: Individual processing units that perform a single step (for example, reading, splitting, format conversion, filtering, embedding, captioning, writing).
- **Tasks**: The unit of data that flows through a pipeline (for video, `VideoTask` holding a `Video` and its `Clip` objects).
- **Executors**: Components that run pipelines on a backend (Ray) with automatic scaling.

![Stages and Pipelines](./_images/stages-pipelines-diagram.png)

## Pipelines

A pipeline orchestrates stages into an end-to-end workflow. Key characteristics:

- **Stage Sequence**: Stages must follow a logical order where each stage's output feeds into the next
- **Input Configuration**: Specifies the data source location
- **Stage Configuration**: Stages accept their own parameters, including model paths and algorithm settings
- **Execution Mode**: Supports streaming and batch processing through the executor

## Stages

A stage represents a single step in your data curation workflow. Video stages are organized into several functional categories:

- **Input/Output**: Read video files and write processed outputs to storage ([Save & Export Documentation](video-save-export))
- **Video Clipping**: Split videos into clips using fixed stride or scene-change detection ([Video Clipping Documentation](video-process-clipping))
- **Frame Extraction**: Extract frames from videos or clips for analysis and embeddings ([Frame Extraction Documentation](video-process-frame-extraction))
- **Embedding Generation**: Generate clip-level embeddings using InternVideo2 or Cosmos-Embed1 models ([Embeddings Documentation](video-process-embeddings))
- **Filtering**: Filter clips based on motion analysis and aesthetic quality scores ([Filtering Documentation](video-process-filtering))
- **Caption and Preview**: Generate captions and preview images from video clips ([Captions & Preview Documentation](video-process-captions-preview))
- **Deduplication**: Remove near-duplicate clips using embedding-based clustering ([Duplicate Removal Documentation](video-process-dedup))

### Stage Architecture

Each processing stage:

1. Inherits from `ProcessingStage`
2. Declares a stable `name` and `resources: Resources` (CPU cores, GPU memory, optional NVDEC/NVENC, or more than one GPU)
3. Defines `inputs()`/`outputs()` to document required attributes and produced attributes on tasks
4. Implements `setup(worker_metadata)` for model initialization and `process(task)` to transform tasks

This design enables map-style execution with executor-managed fault tolerance and dynamic scaling per stage. Stages can optionally provide `process_batch()` to support vectorized batch processing.

Composite stages provide a user-facing convenience API and decompose into one or more execution stages at build time.

```python
class MyStage(ProcessingStage[X, Y]):
    @property
    def name(self) -> str: ...

    @property
    def resources(self) -> Resources: ...

    def inputs(self) -> tuple[list[str], list[str]]: ...
    def outputs(self) -> tuple[list[str], list[str]]: ...

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None: ...
    def process(self, task: X) -> Y | list[Y]: ...
```

Refer to the stage base and resources definitions in Curator for full details.

### Resource Semantics

`Resources` support both fractional and whole‑GPU semantics:

- `gpu_memory_gb`: Request a fraction of a single GPU by memory; Curator rounds to a fractional GPU share and enforces that `gpu_memory_gb` stays within one device.
- `entire_gpu`: Request an entire GPU regardless of memory (also implies access to NVDEC/NVENC on that device).
- `gpus`: Request more than one GPU for a stage that is multi‑GPU aware.
- `nvdecs` / `nvencs`: Request hardware decode/encode units when needed.

Choose one of `gpu_memory_gb` (single‑GPU fractional) or `gpus` (multi‑GPU). Combining both is not allowed.

## Tasks

Video pipelines operate on task types defined in Curator:

- `VideoTask`: Wraps a single input `Video`
- `Video`: Holds decoded metadata, frames, and lists of `Clip`
- `Clip`: Holds buffers, extracted frames, embeddings, and caption windows

Stages transform tasks stage by stage (for example, `VideoReader` populates `Video`, splitting stages create `Clip` objects, embedding and captioning stages annotate clips, and writer stages persist outputs).

## Executors

Executors run pipelines on a backend. Curator uses [`XennaExecutor`](https://github.com/nvidia-cosmos/cosmos-xenna) to translate `ProcessingStage` definitions into Cosmos-Xenna stage specifications and run them on Ray with automatic scaling. Execution modes include streaming (default) and batch.
