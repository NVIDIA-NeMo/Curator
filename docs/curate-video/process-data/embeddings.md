---
description: "Generate clip-level embeddings using Cosmos-Embed1 or InternVideo2"
categories: ["video-curation"]
tags: ["embeddings", "cosmos-embed1", "internvideo2", "video"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "howto"
modality: "video-only"
---

(video-process-embeddings)=

# Embeddings

Generate clip-level embeddings for search, question answering, filtering, and duplicate removal.

## Use Cases

- Prepare semantic vectors for search, clustering, and near-duplicate detection.
- Score optional text prompts against clip content.
- Enable downstream filtering or retrieval tasks that need clip-level vectors.

## Before You Start

- Create clips upstream. Refer to [Clipping](video-process-clipping).
- Provide frames for embeddings or sample at the required rate. Refer to [Frame Extraction](video-process-frame-extraction).
- Access to model weights on each node (the stages download weights if missing).

---

## Quickstart

Use the pipeline stages or the example script flags to generate clip-level embeddings.

::::{tab-set}

:::{tab-item} Pipeline Stage

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.video.clipping.clip_frame_extraction import ClipFrameExtractionStage
from nemo_curator.utils.decoder_utils import FrameExtractionPolicy, FramePurpose
from nemo_curator.stages.video.embedding.internvideo2 import (
    InternVideo2FrameCreationStage,
    InternVideo2EmbeddingStage,
)

pipe = Pipeline(name="video_embeddings_example")
pipe.add_stage(
    ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        extract_purposes=(FramePurpose.EMBEDDINGS,),
        target_res=(-1, -1),
        verbose=True,
    )
)
pipe.add_stage(InternVideo2FrameCreationStage(model_dir="/models", target_fps=2.0, verbose=True))
pipe.add_stage(InternVideo2EmbeddingStage(model_dir="/models", gpu_memory_gb=20.0, verbose=True))
pipe.run()
```

:::

:::{tab-item} Script Flags

```bash
# InternVideo2
python -m nemo_curator.examples.video.video_split_clip_example \
  ... \
  --generate-embeddings \
  --embedding-algorithm internvideo2 \
  --embedding-gpu-memory-gb 20.0

# Cosmos-Embed1 (224p)
python -m nemo_curator.examples.video.video_split_clip_example \
  ... \
  --generate-embeddings \
  --embedding-algorithm cosmos-embed1-224p \
  --embedding-gpu-memory-gb 20.0
```

:::
::::

## Embedding Options

### Cosmos-Embed1

1. Add `CosmosEmbed1FrameCreationStage` to transform extracted frames into model-ready tensors.

   ```python
   from nemo_curator.stages.video.embedding.cosmos_embed1 import (
       CosmosEmbed1FrameCreationStage,
       CosmosEmbed1EmbeddingStage,
   )

   frames = CosmosEmbed1FrameCreationStage(
       model_dir="/models",
       variant="224p",  # or 336p, 448p
       target_fps=2.0,
       verbose=True,
   )
   ```

2. Add `CosmosEmbed1EmbeddingStage` to generate `clip.cosmos_embed1_embedding` and optional `clip.cosmos_embed1_text_match`.

   ```python
   embed = CosmosEmbed1EmbeddingStage(
       model_dir="/models",
       variant="224p",
       gpu_memory_gb=20.0,
       verbose=True,
   )
   ```

#### Parameters

::::{tab-set}

:::{tab-item} CosmosEmbed1FrameCreationStage

```{list-table} Cosmos-Embed1 frame creation parameters
:header-rows: 1
:widths: 22 12 12 54

* - Parameter
  - Type
  - Default
  - Description
* - `model_dir`
  - str
  - `"models/cosmos_embed1"`
  - Directory for model utilities and configs used to format input frames.
* - `variant`
  - {"224p", "336p", "448p"}
  - `"336p"`
  - Resolution preset that controls the model’s expected input size.
* - `target_fps`
  - float
  - 2.0
  - Source sampling rate used to select frames; may re-extract at higher FPS if needed.
* - `num_cpus`
  - int
  - 3
  - CPU cores used when on-the-fly re-extraction is required.
* - `verbose`
  - bool
  - `False`
  - Log per-clip decisions and re-extraction messages.
```

:::

:::{tab-item} CosmosEmbed1EmbeddingStage

```{list-table} Cosmos-Embed1 embedding parameters
:header-rows: 1
:widths: 22 12 12 54

* - Parameter
  - Type
  - Default
  - Description
* - `model_dir`
  - str
  - `"models/cosmos_embed1"`
  - Directory for model weights; downloaded on each node if missing.
* - `variant`
  - {"224p", "336p", "448p"}
  - `"336p"`
  - Resolution preset used by the model weights.
* - `gpu_memory_gb`
  - int
  - 20
  - Approximate GPU memory reservation per worker.
* - `texts_to_verify`
  - list[str] | None
  - `None`
  - Optional text prompts to score against the clip embedding.
* - `verbose`
  - bool
  - `False`
  - Log setup and per-clip outcomes.
```

:::

::::

#### Outputs

- `clip.cosmos_embed1_frames` → temporary tensors used by the embedding stage
- `clip.cosmos_embed1_embedding` → final clip-level vector (NumPy array)
- Optional: `clip.cosmos_embed1_text_match`

### InternVideo2

1. Add `InternVideo2FrameCreationStage` to transform extracted frames into model-ready tensors.

   ```python
   from nemo_curator.stages.video.embedding.internvideo2 import (
       InternVideo2FrameCreationStage,
       InternVideo2EmbeddingStage,
   )

   frames = InternVideo2FrameCreationStage(
       model_dir="/models",
       target_fps=2.0,
       verbose=True,
   )

   ```

2. Add `InternVideo2EmbeddingStage` to generate `clip.intern_video_2_embedding` and optional `clip.intern_video_2_text_match`.

   ```python
   embed = InternVideo2EmbeddingStage(
       model_dir="/models",
       gpu_memory_gb=20.0,
       verbose=True,
   )
   ```

#### Parameters

::::{tab-set}

:::{tab-item} InternVideo2FrameCreationStage

```{list-table} InternVideo2 frame creation parameters
:header-rows: 1
:widths: 22 12 12 54

* - Parameter
  - Type
  - Default
  - Description
* - `model_dir`
  - str
  - `"InternVideo2"`
  - Directory for model utilities used to format input frames.
* - `target_fps`
  - float
  - 2.0
  - Source sampling rate used to select frames; may re-extract at higher FPS if needed.
* - `verbose`
  - bool
  - `False`
  - Log re-extraction and per-clip messages.
```

:::

:::{tab-item} InternVideo2EmbeddingStage

```{list-table} InternVideo2 embedding parameters
:header-rows: 1
:widths: 22 12 12 54

* - Parameter
  - Type
  - Default
  - Description
* - `model_dir`
  - str
  - `"InternVideo2"`
  - Directory for model weights; downloaded on each node if missing.
* - `gpu_memory_gb`
  - float
  - 10.0
  - Approximate GPU memory reservation per worker.
* - `num_gpus_per_worker`
  - float
  - 1.0
  - GPUs reserved per worker for embedding.
* - `texts_to_verify`
  - list[str] | None
  - `None`
  - Optional text prompts to score against the clip embedding.
* - `verbose`
  - bool
  - `False`
  - Log setup and per-clip outcomes.
```

:::

::::

#### Outputs

- `clip.intern_video_2_frames` → temporary tensors used by the embedding stage
- `clip.intern_video_2_embedding` → final clip-level vector (NumPy array)
- Optional: `clip.intern_video_2_text_match`

## Troubleshooting

- Not enough frames for embeddings: Increase `target_fps` during frame extraction or adjust clip length so that the model receives the required number of frames.
- Out of memory during embedding: Lower `gpu_memory_gb`, reduce batch size if exposed, or use a smaller resolution variant.
- Weights not found on node: Confirm `model_dir` and network access. The stages download weights if missing.

## Next Steps

- Use embeddings for duplicate removal. Refer to [Duplicate Removal](video-process-dedup).
- Generate captions and previews for review workflows. Refer to [Captions & Preview](video-process-captions-preview).
