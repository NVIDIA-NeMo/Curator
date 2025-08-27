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

Create frame sequences and compute clip-level embeddings for search, QA, and duplicate removal.

## Cosmos-Embed1

```python
from ray_curator.stages.video.embedding.cosmos_embed1 import (
    CosmosEmbed1FrameCreationStage,
    CosmosEmbed1EmbeddingStage,
)

frames = CosmosEmbed1FrameCreationStage(
    model_dir="/models",
    variant="224p",  # or 336p, 448p
    target_fps=2.0,
    verbose=True,
)
embed = CosmosEmbed1EmbeddingStage(
    model_dir="/models",
    variant="224p",
    gpu_memory_gb=20.0,
    verbose=True,
)
```

## InternVideo2

```python
from ray_curator.stages.video.embedding.internvideo2 import (
    InternVideo2FrameCreationStage,
    InternVideo2EmbeddingStage,
)

frames = InternVideo2FrameCreationStage(
    model_dir="/models",
    target_fps=2.0,
    verbose=True,
)
embed = InternVideo2EmbeddingStage(
    model_dir="/models",
    gpu_memory_gb=20.0,
    verbose=True,
)
```

<!-- end -->
