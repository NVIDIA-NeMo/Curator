---
description: "Remove near-duplicate clips using semantic clustering and pairwise similarity"
categories: ["video-curation"]
tags: ["deduplication", "semantic", "pairwise", "kmeans", "video"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "howto"
modality: "video-only"
---

(video-process-dedup)=
# Duplicate Removal

Use the generic semantic duplicate-removal stages with clip embeddings written to parquet.

```python
from ray_curator.stages.deduplication.semantic.kmeans import KMeansStage
from ray_curator.stages.deduplication.semantic.pairwise import PairwiseSimilarityStage
from ray_curator.stages.deduplication.semantic.ranking import RankingStage

# Example: cluster embeddings then perform pairwise within clusters
kmeans = KMeansStage(num_clusters=1000)
pairwise = PairwiseSimilarityStage(similarity_threshold=0.9)
rank = RankingStage(method="max-similarity")
```

Input format: parquet with columns `id` and `embedding` (produced by the video pipeline’s embedding stages and writer).

<!-- end -->
