---
description: "Beginner-friendly tutorial for running your first Ray-based video splitting pipeline using the Python example"
categories: ["video-curation"]
tags: ["beginner", "tutorial", "quickstart", "pipeline", "video-processing", "ray", "python"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "video-only"

---

(video-tutorials-beginner)=
# Create a Video Pipeline

Learn the basics of creating a video pipeline in Curator by following a split-and-clip pipeline example.

```{contents} Tutorial Steps:
:local:
:depth: 2
```

## Before You Start

- Follow the [Get Started guide](gs-video) to install the package, prepare the model directory, and set up your data paths.

### Concepts and Mental Model

Use this overview to understand how stages pass data through the pipeline.

```{mermaid}
flowchart LR
  V[Videos] --> R[VideoReader]
  R --> S1[Split into clips]
  S1 --> T[Encode/Transcode]
  T --> F[Frame extraction]
  F --> E[Embeddings]
  T --> W[Write clips/metadata]
  E --> W
  classDef dim fill:#f6f8fa,stroke:#d0d7de,color:#24292f;
  class R,S1,T,F,E,W dim;
```

- **Pipeline**: An ordered list of stages that process data.
- **Stage**: A modular operation (for example, read, split, encode, embed, write).
- **Executor**: Runs the pipeline (Ray/Xenna backend).
- **Data units**: Input videos → clip windows → frames → embeddings + files.
- **Common choices**:
  - **Splitting**: fixed stride vs. scene-change (TransNetV2)
  - **Encoding**: `libopenh264`, `h264_nvenc`, or `libx264`
  - **Embeddings**: InternVideo2 or Cosmos-Embed1
- **Outputs**: Clips (mp4), previews (optional), and parquet embeddings for downstream tasks (such as semantic duplicate removal).

For more information, refer to the [Video Concepts](about-concepts-video) section.

---

## 1. Define Imports and Paths

Import required classes and define paths used throughout the example.

```python
from nemo_curator.pipeline import Pipeline

from nemo_curator.stages.video.io.video_reader import VideoReader
from nemo_curator.stages.video.clipping.clip_extraction_stages import (
    FixedStrideExtractorStage,
    ClipTranscodingStage,
)
from nemo_curator.stages.video.clipping.clip_frame_extraction import (
    ClipFrameExtractionStage,
)
from nemo_curator.utils.decoder_utils import FrameExtractionPolicy, FramePurpose
from nemo_curator.stages.video.embedding.internvideo2 import (
    InternVideo2FrameCreationStage,
    InternVideo2EmbeddingStage,
)
from nemo_curator.stages.video.io.clip_writer import ClipWriterStage

VIDEO_DIR = "/path/to/videos"
MODEL_DIR = "/path/to/models"
OUT_DIR = "/path/to/output_clips"
```

## 2. Create the Pipeline

Instantiate a named pipeline to orchestrate the stages.

```python
pipeline = Pipeline(name="video_splitting", description="Split videos into clips")
```

## 3. Define Stages

Add modular stages to read, split, encode, extract frames, embed, and write outputs.

### Read Input Videos

Read videos from storage and extract metadata to prepare for clipping.

```python
pipeline.add_stage(
    VideoReader(input_video_path=VIDEO_DIR, video_limit=-1, verbose=True)
)
```

### Split into Clips

[Create clip windows](video-process-clipping) using fixed intervals or scene-change detection.

::::{tab-set}

:::{tab-item} Fixed stride

```python
pipeline.add_stage(
    FixedStrideExtractorStage(
        clip_len_s=10.0,
        clip_stride_s=10.0,
        min_clip_length_s=2.0,
        limit_clips=0,
    )
)
```

:::

:::{tab-item} TransNetV2 (scene change)

```python
from nemo_curator.stages.video.clipping.video_frame_extraction import VideoFrameExtractionStage
from nemo_curator.stages.video.clipping.transnetv2_extraction import TransNetV2ClipExtractionStage

pipeline.add_stage(VideoFrameExtractionStage(decoder_mode="pynvc", verbose=True))
pipeline.add_stage(
    TransNetV2ClipExtractionStage(
        model_dir=MODEL_DIR,
        threshold=0.4,
        min_length_s=2.0,
        max_length_s=10.0,
        max_length_mode="stride",
        crop_s=0.5,
        gpu_memory_gb=10.0,
        limit_clips=0,
        verbose=True,
    )
)
```

:::

::::

### Encode Clips

Convert clip buffers to H.264 using the selected encoder and settings. Refer to [Clip Encoding](video-process-transcoding) for encoder choices and NVENC setup.

```python
pipeline.add_stage(
    ClipTranscodingStage(
        num_cpus_per_worker=6.0,
        encoder="libopenh264",
        encoder_threads=1,
        encode_batch_size=16,
        use_hwaccel=False,
        use_input_bit_rate=False,
        num_clips_per_chunk=32,
        verbose=True,
    )
)
```

### Prepare Frames for Embeddings (Optional)

[Extract frames](video-process-frame-extraction) at target rates for downstream embedding models.

```python
pipeline.add_stage(
    ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        extract_purposes=(FramePurpose.EMBEDDINGS,),
        target_res=(-1, -1),  # no resize
        verbose=True,
    )
)
```

### Generate Embeddings (InternVideo2)

Create InternVideo2-ready frames and compute clip-level embeddings.

```python
pipeline.add_stage(
    InternVideo2FrameCreationStage(model_dir=MODEL_DIR, target_fps=2.0, verbose=True)
)
pipeline.add_stage(
    InternVideo2EmbeddingStage(model_dir=MODEL_DIR, gpu_memory_gb=20.0, verbose=True)
)
```

### Write Clips and Metadata

Write clips, embeddings, and metadata to the output directory. Refer to [Save & Export](video-save-export) for a full list of parameters.

::::{tab-set}

:::{tab-item} ClipWriterStage

```python
pipeline.add_stage(
    ClipWriterStage(
        output_path=OUT_DIR,
        input_path=VIDEO_DIR,
        upload_clips=True,
        dry_run=False,
        generate_embeddings=True,
        generate_previews=False,
        generate_captions=False,
        embedding_algorithm="internvideo2",
        caption_models=[],
        enhanced_caption_models=[],
        verbose=True,
    )
)
```

:::

:::{tab-item} CLI

When using the example pipeline module, configure the writer-related flags:

```bash
python -m nemo_curator.examples.video.video_split_clip_example \
  --video-dir "$VIDEO_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-clip-path "$OUT_DIR" \
  --no-upload-clips          # optional: do not write mp4s
  --dry-run                   # optional: write nothing, validate only
  --generate-embeddings      # optional: enable embedding outputs
  --generate-captions        # optional: enable captions JSON
  --generate-previews        # optional: enable .webp previews
```

:::

::::

## 4. Run the Pipeline

Run the configured pipeline using the executor.

```python
pipeline.run()
```
